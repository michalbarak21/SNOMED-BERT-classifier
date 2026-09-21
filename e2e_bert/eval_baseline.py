#!/usr/bin/env python
import os, sys, json, time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / "baseline"))

from e2e_bert.data import load_corpus, split_corpus  # noqa: E402

# what the baseline calls an assertion -> what we call a presence label
ASSERTION_TO_PRESENCE = {
    "Present": "present",
    "Absent": "absent", 
    "Hypothetical": "hypothetical",
    "Historical": "historical",
    "Conditional": "conditional",
    "Associated with someone else": "someone_else",	
    # we dropped Possible from the corpus and use conditional insttead. leaving
    # it mapped just marks hedged text wrong, which is what we want.
    "Possible": "possible",
}
# empty now, ConText can reach every label. main() only prints a note if it isnt. 
UNREACHABLE_PRESENCE = set()

PROBLEM_LABELS = {"ADR", "Symptom", "Disease", "Finding"}

# the only two patterns where the mention is one contiguous chunk. everything
# else splits the mention up and the baseline's spans cant express that.
CONTIGUOUS_PATTERNS = {"vanilla", "descriptive"} 


def restrict_candidates(pipe, corpus): 
    """throws out the CADEC dictionary, leaves our 10 concepts""" 
    pipe.snomed_ids =  list(corpus.codes)
    terms= []
    for c in corpus.codes:
        terms.append(corpus.code_to_term[c]) 
    pipe.snomed_terms = terms
    pipe.snomed_emb = pipe._sap_encode(pipe.snomed_terms)	


def span_predictions(pipe, text):
    """one text in, one (code, presence) guess per span out"""
    spans =  []
    for e in pipe.run(text, topk=1)["entities"]:
        a= e.get("assertion")
        if a is None:
            # drug spans come back with no assertion, so ask stage 3 anyway. 
            a = pipe.assertion(text, e["start"], e["end"])
        spans.append({ 
            "mention": e["text"],
            "label": e["label"], 
            "ner_score": e["ner_score"],
            "sim": e["snomed"][0]["sim"],
            "code": e["snomed"][0]["code"],
            "assertion": a["status"],
            "presence": ASSERTION_TO_PRESENCE.get(a["status"]),
        })
    return spans


def select_span(spans, how):
    # the one span a single-span rule reads its answer off 
    if not spans:
        return None
    problems = [s for s in spans if s["label"] in PROBLEM_LABELS]
    if not problems:
        problems = spans  # only drugs found, so use them
    # "ner" = most confident span, anything else = best normalisation score.
    if how == "ner":
        return max(problems, key=lambda s: s["ner_score"])
    return max(problems, key=lambda s: s["sim"])


def resolve(spans, how, gold_code, gold_presence):
    """returns (code, presence, joint, mention) for one example"""
    # the oracle still needs a normal span to fall back on or the confusion
    # tables have nothing to show when it misses
    if how == "any":
        fallback = select_span(spans, "sim")
    else:
        fallback = select_span(spans, how)
    if fallback is None:
        return None, None, False, None

    if how == "any":	
        gc = any(s["code"] == gold_code for s in spans)
        gp = any(s["presence"] == gold_presence for s in spans)
        code = gold_code if gc else fallback["code"]
        presence = gold_presence if gp else fallback["presence"]
        # joint is stricter on purpose, ONE span has to get both right. we never
        # glue two spans together here
        joint = any(s["code"] == gold_code and s["presence"] == gold_presence for s in spans)
    else:
        code = fallback["code"]
        presence = fallback["presence"]
        joint = code == gold_code and presence == gold_presence
    return code, presence, joint, fallback["mention"]


def score(records, corpus):
    from sklearn.metrics import f1_score

    code_true = [corpus.code_to_id[r["gold_code"]] for r in records]
    pres_true = [corpus.presence_to_id[r["gold_presence"]] for r in records]
    # -1 = "no prediction" (nothing detected, or a code outside the corpus). it
    # can never equal a gold id so it always counts as wrong.
    code_pred = [corpus.code_to_id.get(r["pred_code"], -1) for r in records]
    pres_pred = [corpus.presence_to_id.get(r["pred_presence"], -1) for r in records]

    n =  len(records)
    hit_code = [p == t for p, t in zip(code_pred, code_true)]
    hit_pres = [p == t for p, t in zip(pres_pred, pres_true)]

    res= {}
    res["n"] = n
    res["acc_code"] = sum(hit_code)/ n
    res["acc_presence"] = sum(hit_pres) / n
    # same-span joint: with --select any one span had to get both heads right
    res["acc_joint"] = sum(r["hit_joint"] for r in records) / n
    # loose version, the two heads can come from diffrent spans. same number as
    # acc_joint for sim/ner.
    res["acc_joint_cross_span"] = sum(a and b for a, b in zip(hit_code, hit_pres)) / n 
    res["f1_code"] = float(f1_score(code_true, code_pred, average="macro", zero_division=0,
                                    labels=list(range(corpus.num_codes))))
    res["f1_presence"] = float(f1_score(pres_true, pres_pred, average="macro", zero_division=0,
                                        labels=list(range(corpus.num_presences))))
    # lenient: was the gold label in ANY span, not just the chosen one .
    res["acc_code_any_entity"] = sum(r["gold_code"] in r["all_codes"] for r in records) / n
    res["acc_presence_any_entity"] = sum(r["gold_presence"] in r["all_presences"]
                                         for r in records) / n
    res["no_entity_rate"] = sum(r["n_entities"] == 0 for r in records) / n
    res["mean_entities"] = sum(r["n_entities"] for r in records)/ n
    return res


def per_class_table(records, key_gold, key_pred, classes, title):
    rows = [f"\n--- {title} ---", f"{'class':<40} {'n':>5} {'correct':>8} {'acc':>7}"]
    for c in classes: 
        rs = [r for r in records if r[key_gold] == c]
        if not rs:
            continue  # class isnt in the split at all
        ok = sum(r[key_pred] == c for r in rs)
        rows.append(f"{c:<40} {len(rs):>5} {ok:>8} {ok / len(rs):>7.3f}")
    return "\n".join(rows)

def per_pattern_table(records):
    """accuracy per discontinuity pattern, the one the report plots"""

    def row(name, sub):
        n= len(sub)
        code_acc = sum(r["pred_code"] == r["gold_code"] for r in sub) / n
        pres_acc = sum(r["pred_presence"] == r["gold_presence"] for r in sub) / n
        joint = sum(r["hit_joint"] for r in sub)/ n
        # "reachable" = gold code  turned up in some span, so a miss here is teh
        # NER's fault, not the normalisation's
        reachable = sum(r["gold_code"] in r["all_codes"] for r in sub) / n
        nothing = sum(r["n_entities"] == 0 for r in sub) / n
        return (f"{name:<20} {n:>5} {code_acc:>9.3f} {pres_acc:>9.3f} "
                f"{joint:>7.3f} {reachable:>10.3f} {nothing:>10.3f}")

    patterns= sorted({r["pattern"] for r in records})
    rows = ["\n--- pattern (per gold class) ---", 
            f"{'pattern':<20} {'n':>5} {'code acc':>9} {'presence':>9} "
            f"{'joint':>7} {'reachable':>10} {'no entity':>10}"]
    for p in patterns:
        rows.append(row(p, [r for r in records if r["pattern"] == p]))
    # last line pools all the discontinuous examples instead of averaging the 
    # rows above, else patterns with few examples count too much
    discontinuous = [r for r in records if r["pattern"] not in CONTIGUOUS_PATTERNS]
    if discontinuous and len(discontinuous) < len(records):
        rows.append("-"*73)
        rows.append(row("discontinuous (avg)", discontinuous))
    return "\n".join(rows)


def load_run_metrics(path):
    # the two json files a classifier run directory leaves behind
    path= Path(path)
    with open(path / "metrics.json") as f:
        data = json.load(f) 
    with open(path / "config.json") as f:	
        cfg = json.load(f)
    return cfg, data.get("test", {})


def final_table(rows) -> str:
    head = ( 
        "| system | code acc | presence acc | joint acc | code macro-F1 | presence macro-F1 |\n"
        "|---|---|---|---|---|---|"
    )
    out =  [head]
    for name, m in rows:
        out.append(f"| {name} | {m['acc_code']:.3f} | {m['acc_presence']:.3f} | {m['acc_joint']:.3f} | "
                   f"{m['f1_code']:.3f} | {m['f1_presence']:.3f} |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description="score the 3-stage baseline on the same split the e2e model uses",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dataset.jsonl")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--split-seed", type=int, default = 13, help="must match the training run")
    p.add_argument("--candidates", default="target", choices=["target", "cadec"],
                   help="normalise against this corpus' 10 codes, or the full CADEC dictionary")
    p.add_argument("--select", default="sim", choices=["sim", "ner", "any"],
                   help="which detected span the prediction is read off; 'any' is the "
                        "oracle -- each head is correct if any span carries the gold label")
    p.add_argument("--limit", type = int, default=None, help="score only the first N examples")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--compare", nargs="*", default=[], metavar="RUN_DIR", 
                   help="classifier run directories to include in the final table")
    args = p.parse_args(argv)

    # sim is the default so those runs keep the plain directory name
    sfx = "" if args.select == "sim" else f"-{args.select}"
    out_dir = Path(args.out_dir or f"runs/baseline-{args.candidates}{sfx}")
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus =  load_corpus(args.data)
    examples = split_corpus(corpus, seed=args.split_seed)[args.split]
    if args.limit:
        examples = examples[: args.limit]
    print(f"scoring the baseline on {len(examples)} {args.split} examples "
          f"({args.candidates} candidate space, select={args.select})")

    print("loading baseline models (first run downloads ~1 GB of checkpoints)...", file=sys.stderr)
    from pipeline import Pipeline

    pipe = Pipeline() 
    if args.candidates == "target":
        restrict_candidates(pipe, corpus)

    records= []
    started =  time.time()
    for i, ex in enumerate(examples,1):
        spans = span_predictions(pipe, ex.text)
        code, presence, joint, mention = resolve(spans, args.select, ex.code, ex.presence)
        records.append({
            "uid": ex.uid, 
            "pattern": ex.pattern,
            "gold_code": ex.code,
            "gold_presence": ex.presence, 
            "pred_code": code,
            "pred_presence": presence,
            "hit_joint": joint, 
            "all_codes": [s["code"] for s in spans],
            "all_presences": [s["presence"] for s in spans],
            "n_entities": len(spans),
            "mention": mention,
            # just the interesting fields, the scores only bloat the file 
            "spans": [{"mention": s["mention"], "label": s["label"], "code": s["code"],
                       "assertion": s["assertion"]} for s in spans],
        })
        if i % 25 == 0 or i == len(examples):
            rate = i / (time.time()- started)
            print(f"  {i}/{len(examples)}  ({rate:.1f} ex/s)", flush=True)

    metrics = score(records, corpus)
    print(
        f"\nbaseline | acc code {metrics['acc_code']:.3f}  "
        f"presence {metrics['acc_presence']:.3f}  joint {metrics['acc_joint']:.3f}  "
        f"macro-F1 {metrics['f1_code']:.3f}/{metrics['f1_presence']:.3f}"
    )
    if metrics["acc_joint_cross_span"] != metrics["acc_joint"]:
        print(f"          | joint {metrics['acc_joint']:.3f} requires one span to get both "
              f"heads right; crediting the heads to different spans gives "
              f"{metrics['acc_joint_cross_span']:.3f}")
    print(
        f"detection | no entity found for {metrics['no_entity_rate']:.1%} of texts, "
        f"{metrics['mean_entities']:.2f} entities per text, "
        f"gold reachable from some span: code {metrics['acc_code_any_entity']:.3f}, " 
        f"presence {metrics['acc_presence_any_entity']:.3f}"
    )
    if args.select == "any":
        print("note: --select any is an oracle over the detected spans (it reads the gold "
              "label to pick the span), so these are a ceiling, not a comparable system.")

    print(per_class_table(records, "gold_code", "pred_code", corpus.codes,
                          "SNOMED code (per gold class)"))
    print(per_class_table(records, "gold_presence", "pred_presence",
                          corpus.presences, "presence (per gold class)")) 
    print(per_pattern_table(records))
    dead = [c for c in corpus.presences if c in UNREACHABLE_PRESENCE]
    if dead:
        print(f"\nnote: {', '.join(dead)} has no counterpart in the baseline's "
              f"label set and is unreachable by construction.")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test": metrics, "args": vars(args)}, f, indent=2)
    with open(out_dir / "predictions.jsonl", "w") as f: 
        for r in records:
            f.write(json.dumps(r) + "\n")

    label = f"baseline pipeline ({args.candidates} candidates, select={args.select}"
    label += ", oracle)" if args.select == "any" else ")"
    rows = [(label,metrics)]
    # TODO: this label assumes every --compare run was full fine-tuning
    for run in args.compare:
        cfg, test = load_run_metrics(Path(run))
        if test:
            rows.append((f"{cfg['resolved_base_model']} + full fine-tuning", test))
    print("\n" + final_table(rows))
    print(f"\nwrote {out_dir}/metrics.json and {out_dir}/predictions.jsonl")


if __name__ == "__main__":
    main()
