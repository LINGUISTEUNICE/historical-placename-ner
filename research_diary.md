# Research Diary

### Milestone 1 — Choosing the domain and dataset
Picked historical place-name NER over the alternatives (medical terms,
sign language gloss notation) specifically for data reliability — after
the CMU-MOSI ordeal, "is the data actually downloadable" became a
first-class selection criterion, not an afterthought. Found `topres19th`
via the HIPE-2022 shared task: real, GitHub-hosted, well-documented, with
an actual associated data paper (Ardanuy et al., 2022) to cite.

**Lesson carried over from the last project**: checking data
reachability *before* committing to a paper/dataset saved real time here
— confirmed the whole HIPE-2022-data repo clones instantly via `git
clone`, no research-lab server involved.

### Milestone 2 — Building and validating the data parser
HIPE's TSV format is denser than a typical NER dataset (10 columns,
metadata blocks, IOB tags for both literal and metonymic senses, entity
linking columns) — read the format spec carefully before writing the
parser rather than guessing from a few example lines. Tested immediately
against the real training file: confirmed 309 documents, 5,874 sentences,
and a genuinely interesting label set (LOC/STREET/BUILDING, not just one
generic place category) that I hadn't necessarily expected going in.

### Milestone 3 — Building and unit-testing the evaluation metric
`seqeval` (the standard library for this kind of evaluation) failed to
install in this environment — a build-tooling issue
(`setuptools_scm`/`vcs_versioning`), not something fixable by retrying.
Rather than fight the dependency, implemented the same span-based
precision/recall/F1 metric directly, and — same discipline as the TFN
fusion layer — wrote 8 hand-verified unit tests before trusting it:
perfect predictions, completely wrong predictions, partial span overlaps
(which must NOT count as correct — exact boundary match is the standard),
malformed I- tags without a preceding B- tag, and a hand-computed
precision/recall case worked out by hand first, then checked against the
function's output.

Found and fixed one real bug while writing this: an early draft of the
false-negative counting logic had an accidental double-increment (visible
in a leftover comment in the code before I cleaned it up) — caught by
actually running the test suite rather than eyeballing the logic, which
is exactly the kind of small-but-real bug that's easy to miss without
tests.

### Milestone 4 — Full pipeline validation on real data, without real pretrained weights
Same constraint as the TFN project: this sandbox can reach GitHub (so the
real HIPE data was available) but not Hugging Face Hub (so
`distilbert-base-cased`'s actual pretrained weights were not). Rather
than skip validation entirely, trained a small WordPiece tokenizer from
scratch on the project's own training text (fully offline, using the
`tokenizers` library) and used a small randomly-initialized DistilBERT
config with it — proving the entire pipeline (parsing → tokenization →
label alignment → training loop → custom metric computation) works
correctly end-to-end against real data, even without real pretrained
weights. Confirmed subword splitting behaves as expected (an invented
test word split into 10 subword pieces, exercising the label-alignment
logic's handling of multi-subword words).

**This is a repeatable pattern worth remembering for future projects**:
when a pretrained model can't be downloaded in a given environment, a
tokenizer trained from scratch on your own real data plus a
randomly-initialized model of the same architecture gives a genuine,
non-trivial pipeline validation — much stronger evidence than validating
against a completely random/synthetic dataset would be, since the actual
data-dependent logic (parsing, alignment) is fully exercised.

### What's next
Run the real fine-tuning job with actual `distilbert-base-cased`
pretrained weights (needs an environment that can reach Hugging Face Hub
— i.e., a local machine), compare results honestly against the existing
`Livingwithmachines/toponym-19thC-en` reference model, and write up the
comparison.
