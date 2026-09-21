#!/usr/bin/env python3
import os,sys, json, re, time, random, argparse 
from pathlib import Path	
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

BASE = Path(__file__).resolve().parent.parent   # repo root.
CONCEPTS_FILE =  BASE / "snomed_10_descriptions.txt"
ENV_FILE = BASE/ ".env"
DEFAULT_MODEL = "gpt-5.6-terra"

# the 8 ways of writing the condition. each value is . 
# (title, what the model should do, an example)
PATTERNS = {
    "dialogue": (	
        "Doctor-patient conversation", 
        "Write a short multi-turn exchange (2-4 turns, each turn prefixed with a "
        "speaker label such as 'Patient:' / 'Doctor:'). Split the condition across "
        "turns: one turn carries the sensation/quality, a *different* turn carries "
        "the body site or the qualifier, so neither turn names the condition alone.",
        "Patient: I have a lot of pain since my accident last week / Doctor: Where? / "
        "Patient: in my left arm",
    ),
    "unfolding": (
        "Unfolding case description",
        "Write a short narrative (2-3 sentences) that develops over time: the "
        "anatomical site or triggering event is introduced early, the symptom itself "
        "surfaces only in a later sentence, and the reader must combine them.",
        "I fell from my bike the other day on my left arm. I felt nothing at first "
        "but the next morning I woke up to a severe pain.",
    ),
    "ellipsis": (
        "Ellipsis / gapping",
        "Use ellipsis or gapping: coordinate two complaints so that the head noun of "
        "the condition is elided in one conjunct, or the modifiers are stranded away "
        "from the head noun they modify.",
        "The patient reports a tingling sensation in their neck, and in their left "
        "arm, severe, throbbing pain.",
    ),
    "distant_modifier": (
        "Anatomical separation (distant modifiers)",
        "Separate the symptom word from its anatomical site by a long intervening "
        "clause (passive voice, attribution, hedging material), so the two ends of "
        "the mention are many tokens apart.",
        "A sharp, radiating pain was clearly reported by the patient to be shooting "
        "down her left arm.",
    ),
    "idiomatic": ( 
        "Idiomatic / functional description (zero span)",
        "Describe the condition purely through metaphor, functional impairment or "
        "everyday idiom. No clinical symptom term may appear anywhere in the text — "
        "the concept must be inferable only from the imagery or the lost function.",
        "I can't even lift my coffee cup this morning because my left arm just feels "
        "like dead weight.",
    ),
    "ehr_artifact": (
        "Multi-axial EHR artifact",
        "Write a structured EHR-style record: 2-4 'Field: value' pairs (e.g. "
        "Symptom / Quality / Severity / Site / Duration), separated by commas, "
        "newlines or pipes, with the condition's components distributed across "
        "different fields.",
        "Symptom: Throbbing ache, Severity: 8/10, Site: Left upper extremity",
    ),
    "vanilla": ( 
        "Contiguous, condition named as a noun (control)",
        "Write natural free text in which the condition is stated outright inside one "
        "single contiguous span, and that span contains a medical-condition NOUN (or "
        "noun phrase) naming it — the textbook span a BIO tagger is built to extract. "
        "Vary register across examples (patient forum post, triage note, nurse "
        "handover, discharge summary line).",
        "I have a sore throat that started yesterday. / She was treated for "
        "tonsillitis a few winters ago.",
    ),
    "descriptive": (
        "Contiguous, condition described not named (control)",
        "Write natural free text in which the condition is expressed by one single "
        "contiguous span of adjacent words — but that span DESCRIBES the problem "
        "rather than naming it: no condition noun anywhere, only plain everyday "
        "wording, typically a verb or adjective carrying the sensation right next to "
        "the body site it affects.",
        "My throat burns every single time I swallow.",
    ), 
}

# the 2 control patterns - only onees where we dont want the mention split up,
# so they get thier own instructions. key in here = contiguous.
CONTIGUITY_REQUIREMENT = {
    "vanilla": """
CONTIGUITY REQUIREMENT — this is a contiguous control pattern
The opposite of a discontinuity constraint applies. ONE single unbroken run of
words must state the condition, and that run must contain a medical-condition
NOUN naming it (the preferred term, one of the synonyms, or an everyday noun
for the same thing such as "a headache", "muscle pain", "cramps in my calf").
Test each example before you return it: you must be able to highlight one
contiguous span that already names the condition. If you cannot, rewrite it.""",
    "descriptive": """
CONTIGUITY REQUIREMENT — this is a contiguous control pattern
ONE single unbroken run of adjacent words must express the whole condition —
the sensation and the body site together, with nothing intervening, no speaker
change and no field boundary. But that run must NOT *name* the condition. The
preferred term, its synonyms, and any bare condition NOUN ("pain", "an ache",
"soreness", "fatigue", "a cramp", "weakness", "depression", "wind", "gas") are
FORBIDDEN anywhere in the text. Describe it instead in plain everyday words,
carrying the sensation on a VERB or an ADJECTIVE — those are allowed and are
the point ("my back muscles hurt", "I feel worn out all the time", "my knees
are stiff and sore whenever I climb the stairs").
This is not metaphor and not a riddle: the description must be literal and
immediately understandable. Test each example twice before returning it:
(1) can you highlight ONE unbroken span that carries the whole complaint?
(2) does the text avoid every condition noun? Both must be yes.""",
}

# presence modifiers - is it there, denied, past, etc
PRESENCE = {
    "present": (
        "Present", 
        "The condition is currently affirmed for the patient themself.",
    ),
    "absent": (
        "Absent",
        "The condition is explicitly negated or denied for the patient (it is ruled "
        "out or reported as not occurring). The concept must still be clearly "
        "evoked, only under negation.",
    ),
    "conditional": (
        "Conditional",
        "The condition is real and does occur, but only under a specific triggering "
        "circumstance that the text must name — on exertion, when climbing stairs, "
        "after certain foods, whenever a particular drug is taken, in cold weather. "
        "It is neither hedged nor unrealised: the trigger has actually brought it on "
        "before, so state it as a standing fact — the shape of 'breathless on "
        "exertion', 'penicillin brings her out in a rash', 'his eyes stream whenever "
        "he is near a cat', though never those conditions or that wording — and never "
        "as an 'if' about a future that has not happened.",
    ),
    "historical": (
        "Historical",
        "The condition belongs to the patient's past and is not current — resolved, "
        "years ago, 'used to have'.",
    ),
    "hypothetical": (
        "Hypothetical",
        "The condition has not happened; it is raised only as an unrealised future "
        "possibility — safety netting, 'if you develop...', 'should you notice...', "
        "'come back in case it starts'. Do not tie it to a trigger that has already "
        "brought it on: recurring, trigger-bound complaints are the *conditional* "
        "modifier, not this one.",
    ), 
    "someone_else": (
        "Someone else (experiencer is not the patient)",
        "The condition belongs to another person — a family member, carer or "
        "acquaintance — not to the patient/author.",
    ),
} 

# per concept: (what a reader has to see, what it must not be confused with).
# without the 2nd part the model drifts to the neighbour. not medically checked! 
LAY_HINTS = {
    "68962001": (  # Myalgia
        "the ache/soreness must clearly sit in the MUSCLES themselves — the fleshy "
        "parts: thighs, calves, upper arms, shoulders, back muscles — and be "
        "described in everyday words (aching, sore, tender to press, stiff and "
        "achy after exertion)",
        "joint pain (do not put the pain in knees, knuckles, wrists, hips or "
        "'my joints'); muscle cramp (no sudden seizing, knotting or spasm); "
        "plain tiredness or weakness with no pain",
    ),
    "57676002": (  # Arthralgia
        "the pain must clearly sit in the JOINTS — knees, knuckles, wrists, "
        "elbows, hips — where two bones meet, often with stiffness on moving them",
        "muscle pain (keep the pain out of the fleshy muscle bellies); "
        "generalised body aches",
    ),
    "84229001": (  # Fatigue
        "the person must clearly be out of ENERGY — worn out, drained, needing to "
        "rest, everything takes effort",
        "muscle weakness or lack of strength (asthenia); low mood or depression; "
        "sleepiness alone",
    ),
    "55300003": (  # Muscle cramp
        "the muscle must clearly SEIZE UP suddenly — knotting, gripping, locking, "
        "having to stretch it out, often at night or mid-exercise",
        "ordinary muscle aching or soreness (myalgia); joint pain", 
    ),
    "35489007": (  # Depression
        "the person's MOOD must clearly be low — flat, hopeless, no interest or "
        "pleasure in things they used to enjoy, tearful",
        "tiredness or fatigue alone; anxiety; ordinary sadness at a specific event",
    ),
    "162076009": (  # Excessive upper gastrointestinal gas.
        "the gas must clearly be in the UPPER gut — burping, belching, trapped "
        "wind under the ribs, bloated and full after small meals",
        "lower-gut wind or flatulence; heartburn or reflux; stomach pain",
    ),
    "25064002": (  # Headache.
        "the pain must clearly be in the HEAD — forehead, temples, behind the "
        "eyes, band around the skull",
        "neck pain; facial or sinus pain; dizziness",
    ),
    "13791008": (  # Asthenia 
        "the person must clearly have no STRENGTH — legs give way, can't lift or "
        "grip things, needs help standing",
        "tiredness or low energy without weakness (fatigue); muscle pain; "
        "one-sided weakness suggesting a stroke",
    ), 
    "10601006": (  # Pain in lower limb
        "the pain must clearly be somewhere in the LEG as a whole — thigh, shin, "
        "calf, down the leg — without being pinned to one tissue type",
        "back pain; pain in the arms; a pain described specifically as joint pain", 
    ), 
    "161891005": (  # Backache
        "the pain must clearly be in the BACK — lower back, between the shoulder "
        "blades, along the spine",
        "leg pain; neck pain; abdominal pain",
    ),
}

SHARED_SYSTEM_PROMPT = ( 
    "You generate synthetic clinical training text for a research dataset. "
    "PURPOSE: the dataset trains an end-to-end classifier that maps a whole text "
    "to SNOMED CT concepts WITHOUT a span-extraction (NER) step. Standard "
    "pipelines first extract one contiguous span naming the condition, then "
    "classify that span — they fail whenever the condition is expressed across "
    "separate, non-adjacent pieces of text. "
)
STYLE_SYSTEM_PROMPT = (
    " You write realistic, varied English: patient forum posts, triage notes, "
    "consultation transcripts, EHR fragments. You never add commentary, "
    "numbering, quotation marks or explanations around the examples."
)
DISCONTINUOUS_SYSTEM_PROMPT = (
    "Your job on this request is to produce exactly those hard cases: text where "
    "the condition is real and recognisable, but NO single contiguous span of the "
    "text states it. That discontinuity is the entire point of the dataset."
) 
CONTIGUOUS_SYSTEM_PROMPT = (
    "This request is for a CONTROL setting instead. The corpus needs contiguous "
    "text as well — the easy case those pipelines were designed for — so that the "
    "hard cases can be measured against it. Here the condition must be carried by "
    "one single unbroken span of the text, exactly as the user message specifies. "
    "Do not break the mention apart on this request."
)


# how much of the concept's own vocab an example can reuse. None = default for 
# the discontinuous ones, vanilla/descriptive are the two extremes.
NAMING_CLAUSE = {
    None:
        "Exactly one of the {n} may be a near-paraphrase of a synonym; the rest "
        "should\n  require inference rather than string matching — but inference a "
        "layperson can\n  actually make, never a clinical judgement call.",
    "vanilla":
        "Every one of the {n} must name the condition outright, but vary *which* "
        "name:\n  the preferred term, different synonyms, and the everyday lay name "
        "for the same\n  thing. Do not use the same noun phrase twice.",
    "descriptive":
        "None of the {n} may name the condition — each one describes it instead. "
        "Vary\n  the wording of the sensation and the body site so the {n} do not "
        "read as edits\n  of a single sentence.",
} 


def system_prompt(pattern_key) -> str:
    # middle chunk of the system msg is diffrent for the control patterns
    if pattern_key in CONTIGUITY_REQUIREMENT: 
        stance= CONTIGUOUS_SYSTEM_PROMPT
    else: 
        stance =  DISCONTINUOUS_SYSTEM_PROMPT 
    return SHARED_SYSTEM_PROMPT + stance + STYLE_SYSTEM_PROMPT


def build_user_prompt(concept, pattern_key, presence_key, n):
    """the user message for one request (n examples)""" 
    p_title, p_instr,p_example = PATTERNS[pattern_key]
    m_title, m_instr = PRESENCE[presence_key]

    # max 8 synonyms, if the concept has none just reuse the term
    syns =  ", ".join(concept["synonyms"][:8])
    if not syns:
        syns = concept["term"]

    cc = concept["code"]
    if cc in LAY_HINTS:
        must_show, not_mistakable = LAY_HINTS[cc] 
    else:
        # fallbaack for the codes I never wrote a hint for .
        must_show = f"the text must clearly describe {concept['term']} in everyday words"
        not_mistakable = "any other condition"

    naming = NAMING_CLAUSE.get(pattern_key,NAMING_CLAUSE[None])
    naming = naming.format(n = n)

    if pattern_key in CONTIGUITY_REQUIREMENT:
        discontinuity = CONTIGUITY_REQUIREMENT[pattern_key]
    else:
        discontinuity = """
DISCONTINUITY REQUIREMENT — the reason this dataset exists
No single contiguous span of the text may express the whole concept. A reader
must have to combine at least two separate, non-adjacent pieces of the text to
recover it: typically the SENSATION in one place and the BODY LOCATION in
another, held apart by other material, a speaker change or a field boundary.
Test each example before you return it: if you can highlight one unbroken run of
words that already states the condition, rewrite it."""
    return f"""TARGET SNOMED CT CONCEPT
  Code            : {concept["code"]}
  Preferred term  : {concept["term"]}
  Fully specified : {concept["fsn"]}
  Synonyms        : {syns}

LAY CLARITY — the hardest requirement, do not compromise it
An ordinary person with NO medical training must read the text and be able to
say what the problem is. Concretely: {must_show}.
It must NOT be mistakable for {not_mistakable}.
Vague complaints ("everything hurts", "I feel awful", "I'm sore all over") are
FORBIDDEN — they do not identify the concept. Somewhere in the text, in plain
language, both of these must be recoverable:
  1. WHAT the sensation is (aching, burning, gripping, drained, low mood, ...)
  2. WHERE it is / what it affects (the muscles of the thighs, the knees, ...)
Do not have a speaker raise a competing condition and leave it unresolved: if a
doctor asks "joints or muscles?", the answer must settle it.
{discontinuity}

PATTERN — {p_title}
{p_instr}
Illustrative example of the *pattern* (different condition, do not imitate its
wording or its body site): "{p_example}"

PRESENCE MODIFIER — {m_title}
{m_instr}

TASK
Write exactly {n} examples of text that (a) unmistakably describe the target
concept to a layperson, (b) follow the pattern and the requirement above it,
and (c) carry the presence modifier.

REQUIREMENTS
- Each example is self-contained text of 1-4 sentences (or 2-4 dialogue turns /
  EHR fields where the pattern calls for it). Use "\\n" for internal line breaks.
- The {n} examples must differ from each other in wording, body site, register,
  speaker and length. Do not reuse the same opening twice.
- {naming}
- Never write the SNOMED code, the term "SNOMED", or any meta-commentary.
- Do not mention any real person, clinician or institution by name."""


RESPONSE_SCHEMA = {
    "name": "synthetic_examples",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": { 
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False, 
                },
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    },
}


HEADER_RE = re.compile(r"SNOMED CT (\d+)")
FIELD_RE = re.compile(r"^ {2}(\S.*?)\s*:\s*(.*)$")


def load_concepts(path):
    """reads snomed_10_descriptions.txt -> {code: {code, term, fsn, synonyms}}"""
    concepts = {}
    cur = None     # the concept we are filling in right now
    fld = None   # last field name we saw, needed for multi line synonyms
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADER_RE.search(line)
        if m:
            # header line = a new concept starts here
            cur = {"code": m.group(1), "term": "", "fsn": "", "synonyms": []}
            concepts[cur["code"]] = cur
            fld = None
            continue
        if cur is None:
            continue   # junk before the first header
        m = FIELD_RE.match(line)
        if m:
            fld = m.group(1)
            val = m.group(2)
            if fld.startswith("Preferred term"):
                cur["term"] = val
            elif fld.startswith("Fully Specified Name"):
                cur["fsn"] =  val 
            elif fld.startswith("Synonyms") and val:
                cur["synonyms"].append(val)
        elif line.startswith("      ") and fld and fld.startswith("Synonyms"):
            # extra synonyms sit indented deeper after the "Synonyms :" line 
            cur["synonyms"].append(line.strip())
    return concepts


def load_env_file(path=ENV_FILE):
    """tiny .env reader so we dont need python-dotenv"""	
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: 
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip("'\"")   # strip the quotes if there are any
        # setdefault and not [] on purpose, a real env var beats the file
        os.environ.setdefault(key.strip(),val)


def call_openai(client, model, temperature, system, user, retries=4):
    # (parsed json, usage), or (None, None) if every retry failed
    last = None
    for attempt in range(retries):
        try: 
            kw = dict(
                model=model, 
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema","json_schema": RESPONSE_SCHEMA},
            )
            # some models refuse the temperature arugment so only send it if asked
            if temperature is not None:
                kw["temperature"] = temperature
            resp = client.chat.completions.create(**kw)
            return json.loads(resp.choices[0].message.content), resp.usage 
        except Exception as e:
            # catch  everything, the api breaks in a lot of ways and we retry all 
            # TODO: rate limits should probably wait longer than the rest
            last = e
            if attempt == retries-1:
                break
            # back off, plus noise so the workers dont all come back at once .
            time.sleep(2**attempt + random.random())
    print(f"  ! request failed after {retries} attempts: {last}", file=sys.stderr)
    return None, None


def main():
    ap = argparse.ArgumentParser(description="build the synthetic dataset by asking an openai model for texts",
                                 formatter_class=argparse.RawDescriptionHelpFormatter) 
    ap.add_argument("--codes", nargs="+", default=["68962001"],
                    help="SNOMED codes to generate for (default: Myalgia)") 
    ap.add_argument("--patterns", nargs="+", default=list(PATTERNS),
                    choices=list(PATTERNS), help="pattern pool to sample from")
    ap.add_argument("--presence", nargs="+", default=["present"], 
                    choices=list(PRESENCE), help="presence modifiers to sample from")
    ap.add_argument("--batches", type = int, default=14,
                    help="number of LLM requests (each yields --per-batch examples)") 
    ap.add_argument("--per-batch", type=int, default=5, help="examples per request")
    # --balanced is already the default, --iid is what turns it off
    ap.add_argument("--balanced", action="store_true", default=True,
                    help="cycle the (pattern, presence) grid instead of sampling i.i.d.")
    ap.add_argument("--iid", dest="balanced", action="store_false",
                    help="sample (pattern, presence) uniformly at random instead") 
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omitted by default (recent models reject the param); "
                         "set e.g. 1.2 on models that accept it, for more diversity")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")
    ap.add_argument("--seed", type=int,default=0)
    ap.add_argument("--out-dir", default=str(BASE/"data"),
                    help="one JSON file per example is written under this directory") 
    ap.add_argument("--dry-run", action="store_true", 
                    help="print the prompts that would be sent and exit") 
    args = ap.parse_args()

    concepts = load_concepts(CONCEPTS_FILE)
    missing = [c for c in args.codes if c  not in concepts]
    if missing:
        sys.exit(f"codes not found in {CONCEPTS_FILE.name}: {', '.join(missing)}")

    rng =  random.Random(args.seed)

    # every (code, pattern, presence) combo we are allowed to use
    combos = []
    for c in args.codes:
        for p in args.patterns:
            for m in args.presence: 
                combos.append((c, p,m))

    if args.balanced:
        # walk the combos so each one gets about the same number of batches,
        # then shuffle so the workers dont do them in a fixed order
        jobs = []
        for i in range(args.batches):
            jobs.append(combos[i % len(combos)])
        rng.shuffle(jobs) 
    else:
        jobs = [rng.choice(combos) for _ in range(args.batches)]

    if args.dry_run:
        bar = "="*78 
        for i, (code, pat, mod) in enumerate(jobs, 1):
            print(f"\n{bar}\nBATCH {i}  |  {code} {concepts[code]['term']}  |  pattern={pat}  presence={mod}\n{bar}") 
            print(build_user_prompt(concepts[code], pat, mod, args.per_batch)) 
        print(f"\n[dry run] {len(jobs)} requests, "
              f"{len(jobs) * args.per_batch} examples, model={args.model}")
        return

    # imported down here so --dry-run works without the package installed 
    try:
        from openai import OpenAI
    except ImportError: 
        sys.exit("openai not installed — run: uv pip install openai (inside .venv)")
    load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"OPENAI_API_KEY is not set — put it in {ENV_FILE} or export it")
    client = OpenAI()
    temperature= args.temperature

    out_dir = Path(args.out_dir)

    def run(job):
        # one request -> one json file per example it gave back
        i, (code, pat, mod) = job
        prompt = build_user_prompt(concepts[code], pat, mod, args.per_batch)
        system = system_prompt(pat)
        data, usage = call_openai(client, args.model, temperature, system, prompt)
        if data is None:
            return 0, usage
        # one folder per (code, presence), pattern goes in the file name
        batch_dir = out_dir / f"{code}_{mod}"
        batch_dir.mkdir(parents=True, exist_ok = True)
        written = 0
        for j, ex in enumerate(data.get("examples", [])):
            rec = {
                "id": f"{code}-{mod}-{pat}-{i:04d}-{j}",
                "text": ex.get("text", "").strip(),
                "snomed_code": code,
                "snomed_term": concepts[code]["term"],
                "presence": mod,
                "pattern": pat,
                "model": args.model,
                "batch": i, 
                "example_index": j,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                # keep the prompt in the file so we can see what made this later .
                "prompt": {"system": system, "user": prompt},
            }
            path = batch_dir / f"{pat}_{i:04d}_{j}.json"
            path.write_text(json.dumps(rec, indent=4, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            written+= 1

        msg = f"  batch {i:>3} | {pat:<17} {mod:<13} | {written} files"
        if written != args.per_batch:
            # model sometimes returns fewer examples than we asked for
            msg += " (expected %d)"% args.per_batch
        print(msg)
        return written, usage

    print(f"Generating {len(jobs) * args.per_batch} examples "
          f"({len(jobs)} requests, model={args.model}) -> {out_dir}/")

    # requests are slow  and mostly waiting, so threads are good enough
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(run, enumerate(jobs, 1)))

    total = 0
    tok_in= 0 
    tok_out =  0
    for n, u in res:
        total += n
        if u:   # usage is None when the request failed
            tok_in += u.prompt_tokens	
            tok_out+= u.completion_tokens

    print(f"\nWrote {total} JSON files under {out_dir}/")
    print(f"  tokens : {tok_in} in / {tok_out} out")


if __name__ == "__main__":
    main()
