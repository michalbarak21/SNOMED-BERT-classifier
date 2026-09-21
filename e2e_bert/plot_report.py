#!/usr/bin/env python
import json, argparse
from pathlib import Path
import matplotlib
# has to come before pyplot or matplotlib goes looking for a display and the
# pdf save blows up
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# first two are the bar series, ACCENT_C is the training curve
BASELINE_C, FULL_C,ACCENT_C = "#2a78d6", "#1baf7a", "#eb6834"
# INK = text, INK_SOFT = the greyer lables, GRID = gridlines + spines
INK, INK_SOFT, GRID =  "#0b0b0b", "#52514e", "#d9d8d4"

# 2 contiguous controls first, then the 6 discontinuity patterns.
# bars get drawn in this order so dont reshuffle. .
CONTROL_PATTERNS = ["vanilla","descriptive"]
PATTERN_ORDER = CONTROL_PATTERNS + ["dialogue", "unfolding", "ellipsis",
                                    "distant_modifier", "idiomatic", "ehr_artifact"]

# goes under each group of bars. the \n in the long ones is so they wrap
# instead of running into thier neighbour..
PATTERN_LABEL = {
    "vanilla": "vanilla", "descriptive": "descriptive",
    "dialogue": "dialogue", "unfolding": "unfolding",
    "ellipsis": "ellipsis", "distant_modifier": "distant\nmodifier",
    "idiomatic": "idiomatic", "ehr_artifact": "EHR\nartifact",
} 

# legend has to say whcih span the baseline read its answer off. "any" is an	
# oracle so its a ceiling, not a rule you could actually run.
SELECT_LABEL = {"sim": "Baseline NER pipeline",
                "ner": "Baseline NER pipeline (NER-confidence span)",
                "any": "Baseline NER pipeline (oracle span chooser)"}


def style(ax):
    # same look for every axes in both figures, so its in one place
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth = 0.6) 
    ax.set_axisbelow(True)  # grid behind the bars


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def accuracy_by_pattern(records, ok=lambda r: r["pred_code"] == r["gold_code"]):
    # bucket by pattern, then average each bucket
    d = {}
    for r in records:
        if r["pattern"] not in d:
            d[r["pattern"]] = []
        d[r["pattern"]].append(ok(r)) 
    out = {} 
    for p in d:
        # ok() is True/False so sum() just counts the correct ones
        out[p] = sum(d[p])/ len(d[p])
    return out


def baseline_label(run_dir) -> str:
    with open(run_dir / "metrics.json") as f:
        sel = json.load(f).get("args", {}).get("select", "sim")
    return SELECT_LABEL.get(sel, SELECT_LABEL["sim"])


def fig_patterns(runs, out_path):
    brecs = read_jsonl(runs["baseline"] / "predictions.jsonl")
    baseline = accuracy_by_pattern(brecs)
    full = accuracy_by_pattern(read_jsonl(runs["full"] / "predictions_test.jsonl"))
    n_test = len(brecs)

    # (legend name, accuracies, colour) per bar set
    bars = [(baseline_label(runs["baseline"]), baseline, BASELINE_C),
              ("End-to-end BERT", full, FULL_C)]

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    width,x = 0.34, range(len(PATTERN_ORDER))
    for i, (name, data, color) in enumerate(bars):
        # i=0 goes half a bar left of the tick, i=1 half a bar right
        offs = [xi + (i - 0.5)*width for xi in x]
        vals = [data.get(p, 0.0) for p in PATTERN_ORDER]
        ax.bar(offs, vals, width*0.92, label=name, color=color, linewidth=0)
        # number on top of every bar, the green one is too pale to read off the
        # gridlines on its own
        for xi, v in zip(offs, vals):
            ax.text(xi, v+ 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.2, color=INK_SOFT)

    # the point of the whole figure: contiguous controls left of the line, 
    # discontinuous ones right. group names go under the tick labels.
    split = len(CONTROL_PATTERNS)- 0.5
    ax.axvline(split, color=GRID, linewidth=0.9, zorder=0)
    # x in data coords, y in axes coords, so the -0.20/-0.28 below are measured
    # from the bottom of the axes and not from the data. ugly but works.
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for label, span in [("contiguous controls", (0, split)),
                        ("discontinuous patterns", (split, len(PATTERN_ORDER) - 1))]:
        ax.plot(span, [-0.20, -0.20], transform=tr, color=GRID, linewidth=0.8,
                clip_on=False, zorder=0)
        ax.text(sum(span)/2, -0.28, label, transform=tr, ha="center", va="top",
                fontsize=7, style="italic", color=INK_SOFT)

    ax.set_xticks(list(x))
    ax.set_xticklabels([PATTERN_LABEL[p] for p in PATTERN_ORDER], fontsize=7.5, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25,0.5, 0.75, 1.0])
    ax.set_ylabel("condition code accuracy", fontsize=8, color=INK_SOFT)
    # big pad leaves a gap under the title for the legend so they dont collide
    ax.set_title(f"Condition code accuracy by pattern ({n_test} test examples)",
                 fontsize=9, color=INK, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left",
              bbox_to_anchor=(0.0, 1.005), labelcolor=INK_SOFT)
    style(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def scalar_series(run_dir, tag, required=True):
    """pulls one scalar tag out of the newest tensorboard file for a run"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # one event file per training run in there, and the accumulator glues them
    # all into one series if you point it at the dir. so newest first, first hit.
    cands = sorted((run_dir / "tb").rglob("events.out.tfevents.*"),
                        key=lambda p: p.stat().st_mtime, reverse = True)
    for c in cands:
        ea = EventAccumulator(str(c))
        ea.Reload()
        if tag in ea.Tags()["scalars"]: 
            evs = ea.Scalars(tag) 
            return [e.step for e in evs], [e.value for e in evs]
    if not required:
        return None
    raise SystemExit(f"tag {tag} not found under {run_dir}/tb") 


# one legend for both panels, so colour + dashes only ever mean the split.
# fields: tag prefix, legend name, colour, dashes, marker. 
SPLIT_STYLE = (("train", "training", ACCENT_C, (2.2, 1.6), "o"),
               ("val", "validation", BASELINE_C, (), "s"))


def fig_curves(runs, out_path):
    # accuracy left, loss right. cant share one axes, they need diffrent y scales.
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    # joint acc = both heads right. strictest of the three and the one the best
    # epoch is picked on, so thats the one to plot.
    for split, name, color, dash, marker in SPLIT_STYLE:
        # required=False so older runs without train/acc_joint still plot 
        r = scalar_series(runs["full"], f"{split}/acc_joint", required=False) 
        if r is None:
            continue
        steps, vals = r
        ax_acc.plot(steps, vals, color=color, linewidth=1.7, marker=marker,
                    markersize=3.2, dashes=dash or (None, None),
                    label=f"{name} split")	

    ax_acc.set_ylabel("joint accuracy (both heads correct)",
                      fontsize=8, color=INK_SOFT)
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_title("Joint accuracy per epoch", fontsize=9, color=INK,
                     loc="left", pad=8)
    ax_acc.legend(frameon=False, fontsize=7.5, loc="lower right", 
                  labelcolor=INK_SOFT, handlelength=2.2)

    # same two lines for the loss panel. no legend, the left one alreadt says
    # what the colours mean.
    for split, name, color, dash, marker in SPLIT_STYLE:
        steps, vals = scalar_series(runs["full"], f"{split}/loss")
        ax_loss.plot(steps, vals, color=color, linewidth=1.7, marker=marker,
                     markersize=3.2, dashes=dash or (None, None))	

    # log scale. on a linear one everything after epoch 4 is a flat line on zero
    # and you cant see the val loss bottom out, which is the point of the panel.
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("total loss (code + presence, log scale)",
                       fontsize=8, color=INK_SOFT)
    ax_loss.set_title("Total loss per epoch", fontsize=9, color=INK, loc="left", pad=8)

    # shared bits 
    for ax in (ax_acc, ax_loss):
        ax.set_xlabel("epoch", fontsize=8, color=INK_SOFT)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        style(ax) 

    fig.suptitle("Convergence of the end-to-end classifier", fontsize=9.5, 
                 color=INK, x=0.005, ha="left", y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="draw the two report figures from what the runs already saved",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # default baseline is the oracle run on purpose - scores the baseline at its.
    # ceiling so the gap cant be blammed on span selection
    p.add_argument("--baseline", default="runs/baseline-target-any")
    p.add_argument("--full", default="runs/bert-full")
    p.add_argument("--out-dir", default="report/figures")
    args = p.parse_args(argv)

    runs = {"baseline": Path(args.baseline),"full": Path(args.full)}
    out_dir = Path(args.out_dir) 
    out_dir.mkdir(parents=True , exist_ok=True)

    # fixme later: could be flags, but both figures want the same ones anyway
    plt.rcParams.update({"font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK_SOFT, "figure.dpi": 200})
    fig_patterns(runs, out_dir / "fig_patterns.pdf")
    fig_curves(runs, out_dir / "fig_curves.pdf")


if __name__ == "__main__":
    main()
