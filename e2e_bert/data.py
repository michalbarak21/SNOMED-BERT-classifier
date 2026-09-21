#!/usr/bin/env python
import json, random, sys
from dataclasses import dataclass
from pathlib import Path

# same order as PRESENCE in data_generation/synthetic_gen.py 
PRESENCE_ORDER = [
    "present",
    "absent",
    "conditional",
    "historical",
    "hypothetical",
    "someone_else",
]

# repeated over every config: 3 train, 1 val, 1 test per 5 examples
SPLIT_CYCLE = ("train", "train", "train", "val", "test")	

SPLITS = ("train", "val", "test")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "dataset.jsonl"
# the jsonl only has the code, so terms come from the snomed dump
CONCEPTS_FILE = ROOT / "snomed_10_descriptions.txt"


@dataclass(frozen=True)
class Example:
    uid: str
    text: str
    code: str
    term: str
    presence: str
    pattern: str 

    @property
    def config(self):
        return (self.code, self.presence, self.pattern)


class Corpus:
    """the examples + the label vocabs we build from them"""

    def __init__(self, examples):
        self.examples = examples
        self.codes = sorted({e.code for e in examples}) 

        # our own order, anything unexpected gets stuck at the end
        got = {e.presence for e in examples}
        self.presences = [p for p in PRESENCE_ORDER if p in got] 
        self.presences += sorted(got- set(self.presences))

        self.patterns = sorted({e.pattern for e in examples})

        self.code_to_term = {}
        for e in examples:
            self.code_to_term[e.code] =  e.term	

        # label -> index, the two heads need these
        self.code_to_id = {c: i for i, c in enumerate(self.codes)}
        self.presence_to_id = {p: i for i,p in enumerate(self.presences)}

        self.num_codes = len(self.codes)
        self.num_presences =  len(self.presences)

    def code_labels(self):
        # "code term" per code, goes in the report as target_names
        return [f"{c} {self.code_to_term.get(c, '')}".strip() for c in self.codes] 


def load_terms(concepts_file = CONCEPTS_FILE):
    """code -> preferred term, read out of the snomed descriptions file"""
    from data_generation.synthetic_gen import load_concepts

    concepts_file = Path(concepts_file)
    if not concepts_file.exists():
        # the baseline matches against these strings so dont just return {}
        raise SystemExit(f"missing concept descriptions: {concepts_file}")

    out = {}
    for code, c in load_concepts(concepts_file).items(): 
        out[code]= c["term"]
    return out


def load_corpus(data_file=DEFAULT_DATA):
    """reads the jsonl and returns a Corpus"""
    data_file =  Path(data_file)
    if not data_file.is_file():
        raise SystemExit(f"no corpus file at {data_file}")

    terms = load_terms()
    examples = []
    with open(data_file) as f:
        # no ids in the file so the line number is the uid. blank lines still
        # count here or the uids would shift.
        for ln, line in enumerate(f):
            line = line.strip() 
            if not line:
                continue
            d = json.loads(line)
            text = (d.get("text") or "").strip()
            if not text:
                continue
            code = str(d["snomed_code"])  # its an int in some of the rows. 
            # padded to 6 digits so sorting uids as strings keeps line order
            uid = "%06d"% ln
            examples.append(Example(uid, text, code, terms.get(code, ""),
                                    d["presence"], d.get("pattern", "unknown")))

    if not examples:
        raise SystemExit(f"no examples found in {data_file}")
    return Corpus(examples)


def split_corpus(corpus, seed = 13):
    """splits each (code, presence, pattern) group 3 train / 1 val / 1 test"""
    # bucket the examples by config
    by_cfg = {}
    for ex in corpus.examples:
        if ex.config not in by_cfg:
            by_cfg[ex.config] = []
        by_cfg[ex.config].append(ex) 

    splits = {}
    for s in SPLITS:
        splits[s] = []

    # this sorted() and the sort below are what keep the split reproducible, dont drop them
    for config in sorted(by_cfg):
        g = sorted(by_cfg[config], key = lambda e: e.uid)
        # seed comes from the config name so an example always lands in the
        # same split, also on a diffrent machine
        rng = random.Random(f"{seed}|"+ "|".join(config))
        rng.shuffle(g)
        for i,ex in enumerate(g):
            s = SPLIT_CYCLE[i % len(SPLIT_CYCLE)]
            splits[s].append(ex) 

    # back to line order insetad of shuffled order
    for s in SPLITS:
        splits[s].sort(key=lambda e: e.uid)	
    return splits

# TODO: the 3/1/1 ratio should probably be an arugment and not a constant

def describe_splits(corpus, splits):
    """summary of the split, printed before training starts"""
    cfgs = set()
    for e in corpus.examples:
        cfgs.add(e.config)
    n_configs = len(cfgs)

    lines = []
    lines.append(f"{len(corpus.examples)} examples | {corpus.num_codes} codes x "
                 f"{corpus.num_presences} presence x {len(corpus.patterns)} patterns "
                 f"= {n_configs} configurations")

    for s in SPLITS:
        ex = splits[s]
        covered = len({e.config for e in ex})
        # how many examples each code / each presence got in this split
        codes = {}
        pres = {}
        for e in ex:
            codes[e.code] = codes.get(e.code, 0)+ 1
            pres[e.presence] = pres.get(e.presence, 0) + 1 
        lines.append(
            f"  {s:<5} {len(ex):>5} examples | {covered}/{n_configs} configs | "
            f"code {min(codes.values())}-{max(codes.values())} per class | "
            f"presence {min(pres.values())}-{max(pres.values())} per class"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # sanity check: python -m e2e_bert.data dataset.jsonl
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path= DEFAULT_DATA
    corpus = load_corpus(path) 
    print(describe_splits(corpus, split_corpus(corpus)))
