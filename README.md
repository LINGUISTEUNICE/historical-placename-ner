# Historical Place-Name Recognition in 19th-Century English Newspapers

Fine-tunes a small transformer to recognize place names — general
locations, streets, and buildings — in real OCR'd 19th-century English
newspaper text, using the `topres19th` dataset from the HIPE-2022 shared
task.

## The task and dataset

**Dataset**: `topres19th`, from *"A Dataset for Toponym Resolution in
Nineteenth-Century English Newspapers"* (Ardanuy et al., 2022, Journal of
Open Humanities Data), part of the British Library / Alan Turing
Institute's "Living with Machines" project, released via the CLEF
HIPE-2022 shared task.
Paper: https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.56/
Data: https://github.com/hipe-eval/HIPE-2022-data (topres19th subset)

Real digitized 19th-century English newspaper articles (with realistic OCR
noise), token-level annotated for three entity types:
- **LOC**: general place names (countries, cities, regions)
- **STREET**: street names
- **BUILDING**: named buildings

Sizes: 309 training documents (5,874 sentences, ~124k tokens), 34 dev
documents, 112 test documents. **The raw TSV data files are included
directly in this repo** (total ~7MB, well under GitHub's limits) — a
deliberate choice, after the previous project's experience with a
research-lab data server going permanently offline. Self-containment here
means this project keeps working even if the upstream HIPE-2022-data
repo ever disappears.

## Why this project

Historical named entity recognition sits at a genuine intersection of
digital humanities and NLP: automatically identifying place references in
digitized historical text is directly useful for historians, archivists,
and geographers doing large-scale analysis of historical documents — this
isn't a toy task, it's the same problem real DH infrastructure projects
(like Living with Machines) are built to solve.

## What's implemented

- `data/hipe_parser.py` — parser for the HIPE TSV annotation format,
  splitting long newspaper articles into sentences (using the format's own
  sentence-boundary markers) for tractable training examples
- `data/prepare_dataset.py` — subword tokenization with correct label
  alignment (each word's label propagates only to its first subword token,
  standard NER fine-tuning practice)
- `evaluation/span_metrics.py` — span-based precision/recall/F1 (exact
  boundary + type match), implemented directly rather than via `seqeval`
  (which fails to install in some environments — see note below),
  hand-verified against 8 unit-tested cases in
  `evaluation/test_span_metrics.py`
- `training/train_ner.py` — fine-tuning script using HuggingFace
  `Trainer`, defaults to `distilbert-base-cased`
- `app.py` — Gradio demo app, ready to deploy as a Hugging Face Space

## Status: pipeline fully validated on real data, not yet trained with real pretrained weights

Everything above has been run end-to-end against the **real** topres19th
dataset (not synthetic placeholders) — parsing, tokenization, label
alignment, training loop, and metric computation were all validated
together in one sandbox that could reach GitHub (for the data) but not
Hugging Face Hub (for pretrained model weights), so the pipeline
validation used a small randomly-initialized model with a tokenizer
trained from scratch on the project's own data, rather than
`distilbert-base-cased`'s real pretrained weights.

**This means the code is proven correct end-to-end; what's left is simply
running it with real pretrained weights**, which just needs an environment
that can reach `huggingface.co` (i.e., your own machine).

## Setup and running

```bash
pip install -r requirements.txt

# fine-tune (downloads distilbert-base-cased automatically on first run):
python training/train_ner.py --epochs 5

# quick pipeline check on a small subset first, if you want:
python training/train_ner.py --max_train_examples 100 --epochs 1
```

Run the unit tests any time:
```bash
python evaluation/test_span_metrics.py
```

## Hosting

- **Code**: this repo, on GitHub
- **Trained model**: push to the Hugging Face Hub (`huggingface-cli upload`
  or the `push_to_hub=True` Trainer argument) so others can load it with
  `pipeline("token-classification", model="your-username/model-name")`
  without retraining
- **Interactive demo**: `app.py` is ready to deploy as a Hugging Face
  Space (Gradio SDK) — see the project's writeup for exact steps

## Citation

```
@article{ardanuy2022topres19th,
  title={A Dataset for Toponym Resolution in Nineteenth-Century English Newspapers},
  author={Coll Ardanuy, Mariona and Beavan, David and Beelen, Kaspar and Hosseini, Kasra and Lawrence, Jon and McDonough, Katherine and Nanni, Federico and van Strien, Daniel and Wilson, Daniel C. S.},
  journal={Journal of Open Humanities Data},
  volume={8},
  pages={3},
  year={2022},
  doi={10.5334/johd.56}
}
```

## A useful reference point

The "Living with Machines" project itself published a fine-tuned model for
this exact task on the Hugging Face Hub:
`Livingwithmachines/toponym-19thC-en` (built on their own historical
BERT variant, `bert_1760_1900`, pretrained on 5.1 billion words of
digitized 1760-1900 English books). Once this project has real results,
that's a genuine, citable comparison point — a reproduction landing in
the same neighborhood as an established reference model is a meaningfully
stronger result to report than a number with nothing to compare against.
