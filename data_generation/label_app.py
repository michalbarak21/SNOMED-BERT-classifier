#!/usr/bin/env python3
import os, sys, json
import argparse
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

from data_generation.synthetic_gen import CONCEPTS_FILE, PRESENCE, load_concepts
from e2e_bert.data import DEFAULT_DATA, load_corpus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR =  ROOT/"data"/"annotations" 

# the "none of the above" answers in the radio lists
NO_CONDITION = "__none__"
UNCLEAR_PRESENCE = "__unclear__"


def parse_args():
    # streamlit gives us whatever comes after the "--".
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data-file", default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir",default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--annotator", default="")
    args, _ =  ap.parse_known_args(sys.argv[1:])
    return args


@st.cache_data(show_spinner=False)
def load_samples(data_file):
    """one dict per example, sorted by id"""
    # go through load_corpus so the ids match what the training code uses
    samples= []
    for ex in load_corpus(data_file).examples:
        d = {
            "id": ex.uid,
            "text": ex.text,
            # hash the text too. if the corpus is regenerated the ids change
            # and an old log can still be matched up by text
            "text_sha1": hashlib.sha1(ex.text.encode("utf-8")).hexdigest(),
            "gold_code": ex.code,
            "gold_term": ex.term, 
            "gold_presence": ex.presence,
            "pattern": ex.pattern,
        }
        samples.append(d)
    samples.sort(key = lambda s: s["id"])
    return samples


@st.cache_data(show_spinner=False)
def load_concept_options():
    # (code, term) for the 10 concepts, in the order they sit in the file
    concepts = load_concepts(CONCEPTS_FILE)
    stuff = []
    for c in concepts.values():
        stuff.append((c["code"],c["term"]))
    return stuff


def order_for(samples, annotator):
    """same shuffled order every time for a given annotator"""
    # seeding with the name means a reload dosent reshuffle everything.
    idx =  list(range(len(samples)))
    random.Random(f"seed::{annotator}").shuffle(idx)
    return idx


def log_path(out_dir, annotator) -> Path:
    # these end up in a filename so throw out anything weird .
    safe = ""
    for ch in annotator:
        if ch.isalnum() or ch in "-_":
            safe += ch
        else:
            safe += "_"	
    if not safe:
        safe = "anon" 
    return Path(out_dir)/f"annotations_{safe}.jsonl"


def read_log(path):
    """{sample_id: record}. a later line for the same sample wins"""
    seen = {}
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        line= line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # half written line from a crash, cant do anything with it
        seen[rec["sample_id"]] =  rec
    return seen

def append_log(path, record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # flush+fsync so we only lose the item on screen if streamlit dies
        f.flush()
        os.fsync(f.fileno())


def export_json(path, annotations):
    # same data as the log, one json list next to the jsonl
    out = path.with_suffix(".json")
    rows = sorted(annotations.values(), key = lambda r: r["sample_id"])
    out.write_text(json.dumps(rows, indent=4, ensure_ascii=False) + "\n",
                   encoding="utf-8") 
    return out


def render_feedback(rec): 
    """did the answer you just gave match the generators lables"""
    def line(field, ok, yours, gold):
        if ok:
            return f"**{field}** ✅ {gold}"
        return f"**{field}** ❌ you said *{yours}* — generated as **{gold}**"

    cond_yours = rec["condition_term"] or "none of these / unclear"
    cond_gold= rec["gold_term"] or rec["gold_code"]
    if rec["presence"] in PRESENCE:
        pres_yours = PRESENCE[rec["presence"]][0]
    else:
        pres_yours = "unclear"
    if rec["gold_presence"] in PRESENCE:
        pres_gold = PRESENCE[rec["gold_presence"]][0]
    else:
        pres_gold = rec["gold_presence"]

    body = (f"{line('Condition', rec['condition_correct'], cond_yours, cond_gold)}  \n" 
            f"{line('Presence', rec['presence_correct'], pres_yours, pres_gold)}")
    msg = f"Previous example — {body}"
    if rec["condition_correct"] and rec["presence_correct"]:
        st.success(msg) 
    else:
        st.warning(msg)


def main():
    args = parse_args()
    st.set_page_config(page_title="Synthetic clinical text — labelling",
                       page_icon="🩺", layout="centered")

    concept_options = load_concept_options()	
    presence_options =  list(PRESENCE.items())

    # sidebar 1: who is labelling and where the files live
    with st.sidebar:
        st.header("Session")
        annotator = st.text_input("Annotator", value=args.annotator or "anon", 
                                  help="Names the output file; also seeds the "
                                       "presentation order.")
        data_file = st.text_input("Dataset file", value=args.data_file)
        out_dir = st.text_input("Output dir", value=args.out_dir)
        blind = st.toggle("Blind mode", value=True,
                          help="Hide the generator's labels while annotating.")

    try:
        samples = load_samples(data_file)
    except SystemExit as err:
        # load_corpus does sys.exit if the file is missing or empty
        st.error(str(err))
        st.stop()
    if not samples:
        st.error(f"No examples found in `{data_file}`.")
        st.stop() 

    path = log_path(out_dir, annotator)
    # if annotator or out dir got edietd above we are on a diffrent log now,.
    # so re-read it instead of restarting the app.
    if st.session_state.get("_log_key") != str(path):
        st.session_state["_log_key"] = str(path)
        st.session_state["annotations"] = read_log(path) 
        st.session_state["cursor"] = 0
    annotations = st.session_state["annotations"]

    order = order_for(samples, annotator)
    labelled = sum(1 for s in samples if s["id"] in annotations)

    # sidebar 2: progress + how often we agreed witth the generator
    with st.sidebar:
        st.metric("Labelled", f"{labelled} / {len(samples)}")
        st.progress(labelled/len(samples))
        if labelled:
            cond_hits = 0
            pres_hits= 0
            both =  0
            for r in annotations.values():
                if r.get("condition_correct"):
                    cond_hits += 1
                if r.get("presence_correct"):	
                    pres_hits+= 1
                if r.get("condition_correct") and r.get("presence_correct"):
                    both += 1
            c1,c2 = st.columns(2)
            c1.metric("Condition ✓", f"{cond_hits/labelled:.0%}", f"{cond_hits}/{labelled}")
            c2.metric("Presence ✓", f"{pres_hits/labelled:.0%}", f"{pres_hits}/{labelled}")
            st.caption(f"Both correct: {both}/{labelled} ({both/labelled:.0%}) — "
                       f"agreement with the generator's own labels, not ground truth.")
        st.caption(f"Log: `{path}`")
        if st.button("Export JSON snapshot", use_container_width=True):
            st.success(f"Wrote {export_json(path, annotations).name}")
        st.divider() 
        st.caption("Presence statuses")
        for key, (title, desc) in presence_options:
            st.caption(f"**{title}** — {desc}")

    # pick the example to sohw. cursor walks the shuffled order and skips
    # anything already labelled, unless the user went back  to it ("pinned") 
    cursor = st.session_state.get("cursor", 0)
    if not st.session_state.get("pinned"):
        while cursor < len(order) and samples[order[cursor]]["id"] in annotations:
            cursor += 1
        st.session_state["cursor"] = cursor

    if cursor >= len(order):
        st.balloons()
        st.success(f"All {len(samples)} examples labelled. "
                   f"Annotations are in `{path}`.")
        if st.button("Review from the start"):
            st.session_state["cursor"] = 0 
            st.session_state["pinned"] = True
            st.rerun()
        st.stop()

    sample = samples[order[cursor]]
    prev = annotations.get(sample["id"])

    # verdict on the previous example, if there was one
    fb = st.session_state.pop("feedback", None)
    if fb: 
        render_feedback(fb)

    st.subheader(f"Example {cursor + 1} of {len(samples)}")
    # colour has to be set here as well as the background or the light theme.
    # draws dark text on the black box and you cant read it. ugly but works
    st.markdown(
        f"<div style='background:#000;color:#f5f5f5;"
        f"border-left:4px solid #ff4b4b;"
        f"border-radius:6px;padding:1.4rem 1.6rem;font-size:1.5rem;" 
        f"line-height:1.6;white-space:pre-wrap'>{sample['text']}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"`{sample['id']}`")

    if not blind:
        st.info(f"Generated as **{sample['gold_term']}** ({sample['gold_code']}) / "
                f"**{sample['gold_presence']}** — pattern `{sample['pattern']}`")

    st.divider()

    # the actual question. last option in each lisst is the "cant tell" one.
    codes = [c for c, _ in concept_options]+ [NO_CONDITION] 

    def condition_label(code):
        if code == NO_CONDITION:
            return "None of these / unclear"
        term = dict(concept_options)[code]
        return f"{term}  ·  {code}" 

    presence_keys = [k for k, _ in presence_options] + [UNCLEAR_PRESENCE]

    def presence_label(key):
        if key == UNCLEAR_PRESENCE:
            return "Unclear"
        return PRESENCE[key][0]

    with st.form(key=f"form::{sample['id']}", clear_on_submit=False):
        left, right = st.columns([3, 2]) 
        with left:
            st.markdown("**Medical condition**")
            # preselect if we alreadt have an answer for this one
            cond_index = None
            if prev and prev["condition"] in codes:
                cond_index = codes.index(prev["condition"])
            condition = st.radio(
                "Medical condition", codes, format_func=condition_label,	
                index=cond_index,
                label_visibility="collapsed",
            )
        with right:
            st.markdown("**Presence status**") 
            pres_index = None
            if prev and prev["presence"] in presence_keys:
                pres_index = presence_keys.index(prev["presence"])
            presence = st.radio(
                "Presence status", presence_keys, format_func=presence_label,
                index=pres_index,
                label_visibility="collapsed",
            ) 

        old = prev or {}
        notes = st.text_input("Notes (optional)", value=old.get("notes", ""))
        flag = st.checkbox("Flag as a bad example (ambiguous, malformed, off-target)",	
                           value=old.get("flagged", False))

        c1, c2, c3 = st.columns([2, 1,1])
        submitted = c1.form_submit_button("Save & next ▸", type="primary",
                                          use_container_width=True)
        skipped = c2.form_submit_button("Skip", use_container_width=True)
        back = c3.form_submit_button("◂ Back", use_container_width=True) 

    if submitted:
        if condition is None or presence is None:
            st.warning("Pick both a condition and a presence status.")
            st.stop()
        # one log line: the answer + enough context to use it later without
        # opening the dataset again. 
        term = dict(concept_options).get(condition)
        if condition == NO_CONDITION:
            term = None
        record = {
            "sample_id": sample["id"],
            "text_sha1": sample["text_sha1"],
            "annotator": annotator,
            "condition": condition,
            "condition_term": term,
            "presence": presence,
            "notes": notes.strip(),
            "flagged": flag,
            "gold_code": sample["gold_code"], 
            "gold_term": sample["gold_term"],
            "gold_presence": sample["gold_presence"],
            "condition_correct": condition == sample["gold_code"],
            "presence_correct": presence == sample["gold_presence"],
            "pattern": sample["pattern"],
            "text": sample["text"], 
            "source_file": str(Path(data_file).resolve()),
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        append_log(path, record)
        annotations[sample["id"]] = record
        st.session_state["feedback"] = record
        st.session_state["cursor"] = cursor+ 1
        st.session_state["pinned"] = False
        st.rerun()

    if skipped:
        st.session_state["cursor"] = cursor + 1
        st.session_state["pinned"] = True
        st.rerun()

    if back: 
        st.session_state["cursor"] = max(0, cursor-1)
        st.session_state["pinned"] = True
        st.rerun()


if __name__ == "__main__":
    main()
