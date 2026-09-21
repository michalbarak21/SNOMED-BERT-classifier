#!/usr/bin/env python
import os, sys, json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# without this the e2e_bert imports below blow up when you run the file directly 
sys.path.insert(0 , str(Path(__file__).resolve().parent.parent))

from e2e_bert.data import DEFAULT_DATA, load_corpus, split_corpus
from e2e_bert.model import DualHeadClassifier, load_tokenizer
from e2e_bert.train import EncodedDataset, evaluate, make_collate, pick_device


def group_by_pattern(records):
    d= {} 
    for r in records:
        pat = r["pattern"]
        if pat not in d:
            d[pat] = []
        d[pat].append(r)
    return d	


def main(argv=None):
    p = argparse.ArgumentParser(description="re-score a saved checkpoint on one split and dump the predictions",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir") 
    p.add_argument("--split", default="test", choices=["train", "val","test"])
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as fh:
        cfg = json.load(fh)
    device = pick_device(args.device) 

    # old runs used a diffrent key ("data_dir") and a downloaded checkpoint points
    # at whatever path trained it. same examples anyway, so fall back to the default.
    data_file = Path(cfg.get("data", DEFAULT_DATA))
    if not data_file.exists():
        print(f"config.json points at {data_file}, which is not here — "
              f"using {DEFAULT_DATA} instead")
        data_file = DEFAULT_DATA
    corpus = load_corpus(data_file) 
    examples = split_corpus(corpus, seed=cfg["split_seed"])[args.split]

    tokenizer = load_tokenizer(cfg["base"])
    dataset =  EncodedDataset(examples, tokenizer, corpus, cfg["max_length"])
    loader = DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False,
                        collate_fn=make_collate(tokenizer))

    # samme settings as training or the saved weights wont fit. cpu first, then move.
    model = DualHeadClassifier(cfg["base"], corpus.num_codes, corpus.num_presences,
                               dropout=cfg["dropout"], pooling = cfg["pooling"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.to(device) 

    loss_fn =  torch.nn.CrossEntropyLoss()
    metrics, preds = evaluate(model, loader, device, loss_fn,
                              cfg["code_loss_weight"], cfg["presence_loss_weight"])

    print(f"{run_dir}  [{args.split}]  acc code {metrics['acc_code']:.3f}  "
          f"presence {metrics['acc_presence']:.3f}  joint {metrics['acc_joint']:.3f}")

    # shuffle=False in the loader, so preds line up with examples. just zip.
    records = []
    for ex, cp, ct, pp, pt in zip(examples, preds["code_pred"], preds["code_true"],
                                  preds["presence_pred"], preds["presence_true"]):
        rec = {}
        rec["uid"] = ex.uid
        rec["pattern"] = ex.pattern
        rec["gold_code"] = ex.code
        rec["pred_code"] = corpus.codes[cp]
        rec["gold_presence"] = ex.presence
        rec["pred_presence"] = corpus.presences[pp]
        rec["code_ok"] = cp == ct
        rec["presence_ok"] = pp == pt
        records.append(rec)	

    # same numbers again, split by pattern. sorted() so the rows dont move between runs.
    groups = group_by_pattern(records)
    print(f"\n{'pattern':<20} {'n':>5} {'code':>7} {'presence':>9} {'joint':>7}")
    for pat in sorted(groups):
        bits = groups[pat] 
        n= len(bits)
        code_acc = sum(r["code_ok"] for r in bits) / n
        pres_acc = sum(r["presence_ok"] for r in bits)/ n
        joint_acc = sum(r["code_ok"] and r["presence_ok"] for r in bits) / n
        print(f"{pat:<20} {n:>5} {code_acc:>7.3f} {pres_acc:>9.3f} {joint_acc:>7.3f}")

    # TODO: dump just  the wrong ones to a seperate file, would help error analysis. 
    out = run_dir / f"predictions_{args.split}.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) +"\n")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
