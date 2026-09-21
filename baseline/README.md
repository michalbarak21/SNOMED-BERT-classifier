# Baseline pipeline

A simple, non-neural baseline to compare our model against.

Given one sentence, it finds medical mentions and, for each one, prints:
its span in the text, its SNOMED CT code, and whether it's present, absent,
historical, hypothetical, conditional, or about someone else.

It works in three steps:
1. find the span of text and its type (using a pretrained NER model)
2. match that span to the closest SNOMED CT code (using a pretrained embedding model)
3. decide present/absent/etc. using hand-written rules (no model)

Uses the project-root virtual environment. Run it with:

```bash
.venv/bin/python baseline/pipeline.py "My head and stomach ache"
```
