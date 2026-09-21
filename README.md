# End-to-End SNOMED CT Coding of Clinical Conditions

A dataset and model for medical coding over non-continuous spans.

## Results

Test-set accuracy of our model VS the baseline pipeline:

| System | Condition code | Presence modifier | Joint |
|---|---|---|---|
| Baseline pipeline, single-span rule | 0.360 | 0.465 | 0.188 |
| Baseline pipeline, *oracle* span chooser | 0.621 | 0.606 | 0.323 |
| **End-to-end BERT (full fine-tuning)** | **0.988** | **0.979** | **0.967** |

## Downloads

| What | Where | Size |
|---|---|---|
| Our dataset (2,400 labelled examples) | [`dataset.jsonl`](dataset.jsonl) — in this repo | 645 KB |
| Trained model | [checkpoint](https://github.com/michalbarak21/SNOMED-BERT-classifier/releases/latest) | 418 MB |

The checkpoint is the run reported in the table above. To use it:

```bash
mkdir -p runs/bert-full
cd runs/bert-full
curl -LO https://github.com/michalbarak21/SNOMED-BERT-classifier/releases/latest/download/best.pt
curl -LO https://github.com/michalbarak21/SNOMED-BERT-classifier/releases/latest/download/config.json
curl -LO https://github.com/michalbarak21/SNOMED-BERT-classifier/releases/latest/download/metrics.json
cd ../..

.venv/bin/python e2e_bert/eval_checkpoint.py runs/bert-full
```

That prints the test accuracies and writes `predictions_test.jsonl`. It needs no
training and no GPU. The code it does download the `bert-base-uncased` encoder from
the Hugging Face Hub on first run, so it needs network access.

## Install

Needs uv on the PATH env var.

```bash
uv venv --python 3.12 .venv

# everything: classifier, figures, evaluation, baseline pipeline (CPU-only)
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# only to regenerate the corpus or run the annotation GUI
uv pip install --python .venv/bin/python openai streamlit
```

Regenerating the corpus will require `OPENAI_API_KEY` in a `.env` file at the repo root.

## Reproducing the numbers

**Step 1 is ONLY for the case you wish to re-generate the dataset with new LLM calls**.

```bash
# 1. OPTIONAL — regenerate a corpus from scratch (480 LLM calls, needs OPENAI_API_KEY)
.venv/bin/python data_generation/synthetic_gen.py \
    --codes 10601006 13791008 161891005 162076009 25064002 35489007 55300003 57676002 68962001 84229001 \
    --presence present absent conditional historical hypothetical someone_else \
    --batches 480 
    --per-batch 5 
    --workers 8 
    --out-dir data/full

# 2. train the end-to-end classifier
.venv/bin/python e2e_bert/train.py --base bert --epochs 12 --out-dir runs/bert-full

# 3. dump per-example predictions
.venv/bin/python e2e_bert/eval_checkpoint.py runs/bert-full

# 4. score the baseline, both oracle and max-similarity span
.venv/bin/python e2e_bert/eval_baseline.py --select sim --out-dir runs/baseline-target
.venv/bin/python e2e_bert/eval_baseline.py --select any --out-dir runs/baseline-target-any

```

## Which script produces what

| Script | What it does | Produces |
|---|---|---|
| [`data_generation/synthetic_gen.py`](data_generation/synthetic_gen.py) | One LLM call per <concept, pattern, presence> triplet, asking the LLM to generate 5 examples. | Our 2,400-example dataset. |
| [`data_generation/label_app.py`](data_generation/label_app.py) | Streamlit blind-labelling GUI - Label the dataset examples for condition and presence modifier. | The dataset quality numbers: 93% condition-code, 90% presence  accuracy on 30 sampled examples. |
| [`e2e_bert/data.py`](e2e_bert/data.py) | Loads the datasert `dataset.jsonl` and performs the train/val/test split. | - |
| [`e2e_bert/model.py`](e2e_bert/model.py) | `DualHeadClassifier` — shared encoder → `[CLS]` → two independent linear heads (10 codes, 6 statuses). The  loss is the sum of two cross-entropies. | The model design |
| [`e2e_bert/train.py`](e2e_bert/train.py) | Full fine-tuning loop. Selects the checkpoint that got the best validation **joint** accuracy. | The trained BERT model. |
| [`e2e_bert/eval_checkpoint.py`](e2e_bert/eval_checkpoint.py) | Reloads a trained checkpoint and produces its final results | `predictions_test.jsonl` — file with predictions.. |
| [`e2e_bert/eval_baseline.py`](e2e_bert/eval_baseline.py) | Runs the baseline over the test split. | results for the baseline reported in the report |
| [`baseline/pipeline.py`](baseline/pipeline.py) | An implementastion of the baseline system . | The baseline pipeline |

## The Dataset

There is a single JSON example in each linem which looks like:

```json
{"text": "...", "snomed_code": "68962001", "presence": "present", "pattern": "ellipsis"}
```

**10 SNOMED concepts**

| Code | Preferred term |
|---|---|
| 10601006 | Pain in lower limb |
| 13791008 | Asthenia |
| 161891005 | Backache |
| 162076009 | Excessive upper GI gas |
| 25064002 | Headache |
| 35489007 | Depression |
| 55300003 | Muscle cramp |
| 57676002 | Arthralgia |
| 68962001 | Myalgia |
| 84229001 | Fatigue |

**6 presence modifiers**

- `present`
- `absent`
- `conditional`
- `historical`
- `hypothetical`
- `someone_else`

**8 discontinuity patterns** — six discontinuous, two contiguous controls:

| pattern | |
|---|---|
| `dialogue` | Doctor & patient conversations |
| `unfolding` | Unfolding case description, site first and symptom later |
| `ellipsis` | Skipping the repeated noun in a list |
| `distant_modifier` | Anatomical separation: site and sensation far apart |
| `idiomatic` | A functional description, no span exists |
| `ehr_artifact` | Multi-field EHR (not prose) |
| `vanilla` | *control* — continuous, condition named as a noun |
| `descriptive` | *control* — continuous, condition described but not named |

A *configuration* is a single (code, presence, pattern) triplet: 10 × 6 × 8 = 480, each
with 5 examples. The split happens **inside** each configuration — after a
shuffle the five are split to 3x train, 1x val 1x test — so
every configuration appears in all three splits, validation and test cover all 8
patterns for every label pair.
