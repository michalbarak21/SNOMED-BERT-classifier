#!/usr/bin/env python
import argparse, json, os, random, sys, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# has to be set before transformers gets imported or the tokenizr warns about forking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# so running "python e2e_bert/train.py" from the repo root works too
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e2e_bert.data import Example, load_corpus, split_corpus, describe_splits, SPLITS
from e2e_bert.model import BASE_MODELS, DualHeadClassifier, load_tokenizer, parameter_summary, resolve_base_model


class EncodedDataset(Dataset):
    """tokenize the texts once up front, keep the label ids next to them"""

    def __init__(self, examples, tokenizer, corpus, max_length):
        self.examples = examples
        # padding=False, we pad per batch in collate instead. padding everything. 
        # to max_length here just burns compute.
        tmp = tokenizer([e.text for e in examples], truncation=True, max_length=max_length, padding=False)
        self.encodings = tmp

        # labels stay as ids, the corpus has the mapping
        self.code_labels = [] 
        self.presence_labels = []	
        for e in examples:
            self.code_labels.append(corpus.code_to_id[e.code])
            self.presence_labels.append(corpus.presence_to_id[e.presence])

    def __len__(self): 
        return len(self.examples) 

    def __getitem__(self, i): 
        item = {k: v[i] for k, v in self.encodings.items()}
        item["code_label"] = self.code_labels[i]
        item["presence_label"] = self.presence_labels[i]
        return item 


def make_collate(tokenizer):
    # the fn we hand to the DataLoader. tokenizer.pad() only knows the encoder 
    # fields, so the two lables get pulled out of the dicts and put back after.
    def collate(batch):
        codes = torch.tensor([b.pop("code_label") for b in batch], dtype=torch.long)
        presences = torch.tensor([b.pop("presence_label") for b in batch], dtype=torch.long)
        padded = tokenizer.pad(batch, return_tensors="pt")
        padded["code_label"] = codes
        padded["presence_label"] =  presences	
        return padded

    return collate	


def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    # auto = whatever this box has
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward_batch(model, batch, device):
    # not every tokenizer hands us token_type_ids, the model copes with None
    if "token_type_ids" in batch: 
        tt = batch["token_type_ids"].to(device)
    else:
        tt = None
    return model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        token_type_ids=tt,
    )


def evaluate(model, loader, device, loss_fn, w_code, w_presence):
    """run the model over a loader, gives back (metrics, predictions)"""
    model.eval()
    tots = {"loss": 0.0, "loss_code": 0.0, "loss_presence": 0.0}
    n = 0
    code_pred, code_true, pres_pred, pres_true = [], [], [], [] 

    with torch.no_grad():
        for batch in loader:
            code_logits, pres_logits = forward_batch(model, batch, device)
            y_code = batch["code_label"].to(device)
            y_pres = batch["presence_label"].to(device)
            l_code = loss_fn(code_logits, y_code)
            l_pres = loss_fn(pres_logits, y_pres)

            # losses come back averaged over the batch, so scale by batch size
            # here and divide by n at the end, else the short last batch counts too much
            bs = y_code.size(0)
            tots["loss_code"] += l_code.item() * bs
            tots["loss_presence"] += l_pres.item() * bs
            tots["loss"] += (w_code * l_code + w_presence * l_pres).item() * bs
            n += bs

            code_pred += code_logits.argmax(-1).cpu().tolist()
            pres_pred += pres_logits.argmax(-1).cpu().tolist()
            code_true += y_code.cpu().tolist()
            pres_true += y_pres.cpu().tolist()

    from sklearn.metrics import f1_score

    # nmupy arrays so the cmopares below are elementwise .
    cp_a, ct_a = np.array(code_pred), np.array(code_true)
    pp_a, pt_a = np.array(pres_pred), np.array(pres_true)

    metrics = {}
    for k, v in tots.items():
        metrics[k] = v / max(n, 1)  # max() so an empty split dosent divide by zero
    code_ok = cp_a == ct_a
    pres_ok = pp_a == pt_a
    metrics["acc_code"] = float(code_ok.mean())
    metrics["acc_presence"] = float(pres_ok.mean())
    metrics["acc_joint"] = float((code_ok & pres_ok).mean())  # both heads right
    metrics["f1_code"] = float(f1_score(ct_a, cp_a, average="macro", zero_division=0))
    metrics["f1_presence"] = float(f1_score(pt_a, pp_a, average="macro", zero_division=0)) 

    preds = {
        "code_pred": code_pred,
        "code_true": code_true,
        "presence_pred": pres_pred, 
        "presence_true": pres_true, 
    }
    return metrics, preds


def log_metrics(writer, split, metrics, step):
    for key, value in metrics.items():
        writer.add_scalar(f"{split}/{key}", value, step)


def report(corpus, preds) -> str:
    from sklearn.metrics import classification_report

    # code + its term, the bare numbers are unreadable
    cnames = [] 
    for c in corpus.codes:
        cnames.append(f"{c} {corpus.code_to_term.get(c, '')}".strip())

    out = ["\n--- SNOMED code ---"]
    out.append(
        classification_report(
            preds["code_true"], preds["code_pred"],
            labels=list(range(corpus.num_codes)), target_names=cnames,
            zero_division=0, digits=3,
        )
    )
    out.append("--- presence status ---")
    out.append(
        classification_report(
            preds["presence_true"], preds["presence_pred"],
            labels=list(range(corpus.num_presences)), target_names=corpus.presences,
            zero_division=0, digits=3,
        )
    )
    return "\n".join(out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="train the end to end model, one bert encoder with two heads on top", formatter_class=argparse.RawDescriptionHelpFormatter)

    # data
    p.add_argument("--data", default="dataset.jsonl", help="corpus JSONL (text + labels, one example per line)")
    p.add_argument("--split-seed", type=int, default=13, help="seed for the per-configuration 3-1-1 split")

    # model
    p.add_argument("--base", default="bert", metavar="NAME",
                   help="base encoder: " + " | ".join(BASE_MODELS) + " | any HF model id")
    p.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    p.add_argument("--dropout", type=float, default=0.1)

    # hparams
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--code-loss-weight", type=float, default=1.0)
    p.add_argument("--presence-loss-weight", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # the rest 
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--out-dir", default=None, help="run directory (default runs/<base>-full-<timestamp>)")
    p.add_argument("--log-every", type=int, default=10, help="training steps between TensorBoard loss points")
    p.add_argument("--no-test", action="store_true", help="skip the final test-set evaluation")
    p.add_argument("--no-save", action="store_true", help="do not write model weights")

    args = p.parse_args(argv)
    # no --out-dir given, make one up from the timestamp
    if args.out_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        tag = args.base.replace("/", "-")  # ugly but works.
        args.out_dir = f"runs/{tag}-full-{ts}"
    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    # data.
    corpus = load_corpus(args.data)	
    splits = split_corpus(corpus, seed=args.split_seed)
    print(describe_splits(corpus, splits))

    tokenizer = load_tokenizer(args.base)
    dsets = {}
    for s in SPLITS:
        dsets[s] = EncodedDataset(splits[s], tokenizer, corpus, args.max_length)
    collate = make_collate(tokenizer)
    train_loader = DataLoader(dsets["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=False)
    # no shuffle for val/test so the preds line up with the examples
    eval_loaders = {}
    for s in ("val", "test"):
        eval_loaders[s] = DataLoader(dsets[s], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    # dont move this above the dataloaders! building the layers pulls from the
    # random generator and that shifts everything random after it. 
    model = DualHeadClassifier( 
        base_model=args.base,
        num_codes=corpus.num_codes,
        num_presences=corpus.num_presences,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    model.to(device)
    print(f"\nbase={resolve_base_model(args.base)}  device={device}  lr={args.lr}")
    print(f"parameters: {parameter_summary(model)}")

    # optimizer + schedule
    from transformers import get_linear_schedule_with_warmup 

    # no weight decay on biases and LayerNorm, same as the hf examples
    wd, no_wd = [], []
    for name, param in model.named_parameters(): 
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "LayerNorm" in name:
            no_wd.append(param)
        else: 
            wd.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": wd, "weight_decay": args.weight_decay},
         {"params": no_wd, "weight_decay": 0.0}],
        lr=args.lr,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)
    w_code, w_pres = args.code_loss_weight, args.presence_loss_weight

    # dump the settings next to the weights, eval_checkpoint.py reads them back
    cfg = dict(vars(args))
    cfg["resolved_base_model"] = resolve_base_model(args.base)
    cfg["device"] = str(device)
    cfg["codes"] = corpus.codes
    cfg["code_terms"] = corpus.code_to_term
    cfg["presences"] = corpus.presences
    cfg["split_sizes"] = {}
    for s in SPLITS:
        cfg["split_sizes"][s] = len(splits[s])
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    # training loop
    step = 0
    best = {"acc_joint": -1.0, "epoch": -1}  # -1 so the first epoch always wins
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_tot = {"loss": 0.0, "loss_code": 0.0, "loss_presence": 0.0}
        correct_code = correct_pres = correct_joint = seen = 0

        for batch in train_loader:
            code_logits, pres_logits = forward_batch(model, batch, device)
            y_code = batch["code_label"].to(device)
            y_pres = batch["presence_label"].to(device)
            l_code = loss_fn(code_logits, y_code)
            l_pres = loss_fn(pres_logits, y_pres)
            # both heads trained at once, the weights say which one counts more
            loss = w_code * l_code + w_pres * l_pres 

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
                )
            optimizer.step()
            scheduler.step()
            step += 1

            # totals for the epoch summary line
            bs = y_code.size(0)
            run_tot["loss"] += loss.item() * bs
            run_tot["loss_code"] += l_code.item() * bs
            run_tot["loss_presence"] += l_pres.item()* bs
            hit_code = code_logits.argmax(-1) == y_code 
            hit_pres = pres_logits.argmax(-1) == y_pres
            correct_code += hit_code.sum().item()
            correct_pres += hit_pres.sum().item()
            correct_joint += (hit_code & hit_pres).sum().item()
            seen += bs

            # every log_every steps only, a point per step makes the event files huge
            if step % args.log_every == 0:
                writer.add_scalar("train_step/loss", loss.item(), step)	
                writer.add_scalar("train_step/loss_code", l_code.item(), step)
                writer.add_scalar("train_step/loss_presence", l_pres.item(), step)
                writer.add_scalar("train_step/lr", scheduler.get_last_lr()[0], step)

        train_metrics = {}
        for k, v in run_tot.items():
            train_metrics[k] = v / seen
        train_metrics["acc_code"] = correct_code / seen 
        train_metrics["acc_presence"] = correct_pres / seen
        train_metrics["acc_joint"] = correct_joint / seen

        val_metrics, _ = evaluate(model, eval_loaders["val"], device, loss_fn, w_code, w_pres)
        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, "val", val_metrics, epoch)

        print( 
            f"epoch {epoch:>3}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['acc_code']:.3f}/{train_metrics['acc_presence']:.3f} | " 
            f"val loss {val_metrics['loss']:.4f} " 
            f"acc {val_metrics['acc_code']:.3f}/{val_metrics['acc_presence']:.3f} "
            f"joint {val_metrics['acc_joint']:.3f}"
        )

        # checkpoint the best epoch, joint acc is the number we report
        if val_metrics["acc_joint"] > best["acc_joint"]:
            best = dict(val_metrics)
            best["epoch"] = epoch
            if not args.no_save:
                torch.save(model.state_dict(), out_dir / "best.pt")

    print(f"\nbest epoch {best['epoch']} — val joint accuracy {best['acc_joint']:.3f}")

    # test
    results = {"val_best": best}
    if not args.no_test:
        # the model in memory is the last epoch, so load the best one back first.
        # with --no-save theres nothing to load and the test numbers are the last epoch.
        if not args.no_save and (out_dir / "best.pt").exists():
            model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
        test_metrics, test_preds = evaluate(model, eval_loaders["test"], device, loss_fn, w_code, w_pres)
        log_metrics(writer, "test", test_metrics, best["epoch"])
        results["test"] = test_metrics
        print(
            f"test  | acc code {test_metrics['acc_code']:.3f}  " 
            f"presence {test_metrics['acc_presence']:.3f}  joint {test_metrics['acc_joint']:.3f}  " 
            f"macro-F1 {test_metrics['f1_code']:.3f}/{test_metrics['f1_presence']:.3f}"
        )
        print(report(corpus, test_preds))

    # this is what lets you compare runs in the HPARAMS tab
    # no idea why add_hparams wants two seperate dicts, but it does.
    hparams = {
        "base": resolve_base_model(args.base), "lr": args.lr,
        "batch_size": args.batch_size, "epochs": args.epochs,
    }
    fin = results.get("test") or best  # --no-test, fall back to the val numbers
    hm = {}
    for k, v in fin.items(): 
        if isinstance(v, (int, float)):
            hm[f"hparam/{k}"] = v
    writer.add_hparams(hparams, hm)
    writer.close()

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nrun directory: {out_dir}\ntensorboard --logdir {out_dir.parent}")


if __name__ == "__main__":
    main()
