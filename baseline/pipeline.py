#!/usr/bin/env python
import os, sys, json, argparse, warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE= os.path.dirname(os.path.abspath(__file__))
SNOMED_DICT = os.path.join(HERE,"snomed_dict.json")
TRIGGER_DIR =  os.path.join(HERE, "context_triggers")

# the two hf checkpoints, stage 1 and stage 2
NER_MODEL = "csNoHug/bert-base-cased-finetuned-ner-cadec"
SAP_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" 

# what we count as a "problem", the rest are drugs .
PROBLEM_LABELS = {"ADR", "Symptom", "Disease", "Finding"}

# ConText modifier category -> status we report for it
CAT2ASSERT = {"NEGATED_EXISTENCE": "Absent",
              "FAMILY": "Associated with someone else",
              "HYPOTHETICAL": "Hypothetical",
              "CONDITIONAL": "Conditional",
              "POSSIBLE_EXISTENCE": "Possible",
              "HISTORICAL": "Historical"}

# several modifiers can fire on one mention, first in this list wins. negation
# first on purpose, HISTORICAL beats CONDITIONAL becuase "last pregnancy" = past.
ASSERT_PRIORITY = ["NEGATED_EXISTENCE", "FAMILY", "HYPOTHETICAL", "HISTORICAL", 
                   "CONDITIONAL", "POSSIBLE_EXISTENCE"] 
DEFAULT_ASSERT = "Present"  # nothing fired

# medspaCy ships negation/hedging rules but no pseudo triggers, so "history of
# present illness" wrongly fires HISTORICAL. these are the orig ConText lists.
TRIGGER_FILES = {"experiencer_triggers.txt": "FAMILY",
                 "history_triggers.txt": "HISTORICAL",
                 "hypothetical_triggers.txt": "HYPOTHETICAL",
                 "conditional_triggers.txt": "CONDITIONAL"}

# tag in the trigger files -> (medspaCy direction, max_scope)
TAG2DIRECTION = {"PREN": ("FORWARD", None), "POST": ("BACKWARD", None),
                 "ONEW": ("FORWARD", 1), "PSEU": ("PSEUDO", None)}


def load_trigger_rules(directory=TRIGGER_DIR): 
    """reads the trigger .txt files into medspaCy ConTextRule objects"""
    from medspacy.context import ConTextRule

    rr =  []
    for fname, category in TRIGGER_FILES.items():
        # a conjunction should only close the scope of its own feature, so every
        # feature gets a private pseudo category. "presentts" ends HISTORICAL only..
        cj = f"{category}_CONJ"
        with open(os.path.join(directory, fname)) as f:
            for line in f:
                # line looks like: literal <TAB><TAB> [TAG]
                bits = [p for p in line.strip().split("\t") if p] 
                if len(bits) != 2: 
                    continue  # blank line or junk
                literal, tag = bits[0],bits[1].strip("[]").upper()
                if tag == "CONJ":
                    rr.append(ConTextRule(literal , cj, direction="PSEUDO"))
                elif tag in TAG2DIRECTION:
                    direction, scope =  TAG2DIRECTION[tag]
                    rr.append(ConTextRule(literal, category, direction=direction,
                                             max_scope=scope, terminated_by={cj}))
    return rr


class Pipeline:
    """loads the 3 stages once, then call run() as often as you like"""

    def __init__(self):
        import numpy as np
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
        import spacy, medspacy      # importing medspacy is what registers medspacy_context
        self.np, self.torch = np,torch
        # leftover from when _sap_encode used it, nothing calls it now
        self._no_grad = torch.no_grad

        # satge 1, the NER model. aggregation_strategy="max" gives whole spans back
        ner_tok = AutoTokenizer.from_pretrained(NER_MODEL)
        ner_mdl = AutoModelForTokenClassification.from_pretrained(NER_MODEL).eval()
        self.ner = pipeline("token-classification", model=ner_mdl, tokenizer=ner_tok,
                            aggregation_strategy="max", device = -1)

        # stage 2, SapBERT. embed the whole SNOMED dict once here. slow to start but
        # then  every lookup is one matmul
        self.sap_tok = AutoTokenizer.from_pretrained(SAP_MODEL)
        self.sap_mdl = AutoModel.from_pretrained(SAP_MODEL).eval()
        id2term= json.load(open(SNOMED_DICT)) 
        self.snomed_ids = list(id2term.keys())
        self.snomed_terms = [id2term[i] for i in self.snomed_ids]
        self.snomed_emb = self._sap_encode(self.snomed_terms)

        # stage 3, assertion. no model here, all rules
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("medspacy_context")
        self._ctx = self.nlp.get_pipe("medspacy_context")
        self._ctx.add(load_trigger_rules())

    def _sap_encode(self, texts, bs = 64):
        """embeds a list of strings -> (len(texts), 768) numpy array"""
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i+ bs]
                t = self.sap_tok(batch, padding=True, truncation=True,
                                 max_length=32, return_tensors="pt")
                # for SapBERT you want the CLS vector
                v = self.sap_mdl(**t).last_hidden_state[:, 0, :]
                v = torch.nn.functional.normalize(v, dim = 1)
                out.append(v.cpu().numpy())
        return self.np.vstack(out)

    def normalize(self, mention, topk=3):
        """closest SNOMED entries for a mention, best first"""
        q = self._sap_encode([mention])
        # both sides are normalised alreadt so dot product = cosine.
        sc = (q @ self.snomed_emb.T)[0]
        idx = self.np.argsort(-sc)[:topk]
        res= []
        for j in idx:
            res.append((self.snomed_ids[j], self.snomed_terms[j],float(sc[j])))
        return res

    def _modifiers(self, sentence, start, end):
        # every ConText modifier whose scope reaches our mention
        doc = self.nlp(sentence)
        sp = doc.char_span(start, end, label="PROBLEM", alignment_mode="expand") 
        if sp is None: 
            return []   # offsets didnt land on a token
        doc.ents = [sp]
        self._ctx(doc)
        mods =  [] 
        for m in doc.ents[0]._.modifiers:
            mods.append((m.category, m.rule.literal)) 
        return mods

    def assertion(self, sentence, start, end):
        """assertion status for one mention, ConText rules only"""
        mods = self._modifiers(sentence, start, end)
        cats = [c for c,_ in mods]
        for cat in ASSERT_PRIORITY:
            if cat in cats:
                return dict(status=CAT2ASSERT[cat], decided_by="ConText",
                            category=cat, modifiers=mods) 
        return dict(status=DEFAULT_ASSERT, decided_by="ConText default",	
                    category=None, modifiers=mods)

    def run(self, sentence, topk=3):
        """all 3 stages on one sentence, result is ready for json.dumps"""
        spans = self.ner(sentence)
        results = []
        for e in spans: 
            s, en = int(e["start"]),int(e["end"])
            mention = sentence[s:en]
            cc = []
            for c, t, sim in self.normalize(mention, topk):
                cc.append(dict(code=c, term=t, sim=round(sim,3)))
            item = dict(text=mention, label=e["entity_group"], start=s, end=en,	
                        ner_score=round(float(e["score"]),3), snomed=cc)
            # only problems get an assertion, a drug is just a span + code
            if e["entity_group"] in PROBLEM_LABELS:
                item["assertion"] = self.assertion(sentence, s, en)
            results.append(item)
        return dict(sentence=sentence, entities=results)


# printing + the command line bit 
def fmt(out) -> str:
    n = len(out["entities"])
    word = "entity" if n == 1 else "entities"
    lines = [f'\nSentence: "{out["sentence"]}"',
             f'{n} {word} detected\n']
    if not out["entities"]:
        lines.append("  (no spans detected)") 
    for e in out["entities"]:
        best = e["snomed"][0]
        lines.append(f'  ▸ "{e["text"]}"  [{e["label"]}]  (chars {e["start"]}-{e["end"]}, ner {e["ner_score"]})')
        lines.append(f'      SNOMED CT : {best["code"]} | {best["term"]}   (cos {best["sim"]:.2f})') 
        # the rest go underneath, lined up with the first one. 
        for alt in e["snomed"][1:]:
            lines.append(f'                  alt: {alt["code"]} | {alt["term"]}   (cos {alt["sim"]:.2f})')
        if "assertion" in e:
            a = e["assertion"]
            cue = ", ".join(f'{c}:"{l}"' for c, l in a["modifiers"]) or "—"
            lines.append(f'      Assertion : {a["status"]}   (by {a["decided_by"]}; cues: {cue})')
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Baseline CADEC pipeline on a single sentence.")
    ap.add_argument("sentence", help='sentence to process, or "-" to read one line from stdin')
    ap.add_argument("--topk", type=int, default = 3, help="SNOMED candidates to show (default 3)")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of a report")
    args = ap.parse_args()

    if args.sentence == "-":	
        sentence = sys.stdin.readline().strip()
    else:
        sentence = args.sentence
    if not sentence:
        ap.error("empty sentence")

    # models are big, warn the user so they dont sit there wondering
    print("loading models (first run downloads ~1 GB of checkpoints)...", file=sys.stderr)
    out = Pipeline().run(sentence, topk = args.topk)
    if args.json:
        print(json.dumps(out, indent = 2))
    else:
        print(fmt(out))


if __name__ == "__main__":
    main()
