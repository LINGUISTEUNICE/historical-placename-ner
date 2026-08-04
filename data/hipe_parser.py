"""Parses HIPE-2022 format TSV files (used by the topres19th dataset) into
sentence-level (tokens, labels) pairs suitable for HuggingFace token
classification fine-tuning.

HIPE format reference: https://github.com/hipe-eval/HIPE-2022-data
 - Tab-separated columns, first column TOKEN, second NE-COARSE-LIT (IOB tag)
 - Lines starting with '#' are metadata/comments (one block per document)
 - Empty lines separate documents
 - The MISC column (last) contains 'EndOfSentence' to mark sentence boundaries
   -- topres19th documents are often long (whole newspaper articles), so we
   split into sentences for more tractable training examples.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    metadata: dict = field(default_factory=dict)
    sentences: list = field(default_factory=list)  # list of (tokens, labels)


def parse_hipe_tsv(path: str) -> list[Document]:
    documents: list[Document] = []
    current_doc: Document | None = None
    current_tokens: list[str] = []
    current_labels: list[str] = []

    def flush_sentence():
        nonlocal current_tokens, current_labels
        if current_tokens and current_doc is not None:
            current_doc.sentences.append((current_tokens, current_labels))
        current_tokens = []
        current_labels = []

    def flush_document():
        flush_sentence()
        if current_doc is not None and current_doc.sentences:
            documents.append(current_doc)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                # blank line: end of document
                flush_document()
                current_doc = None
                continue

            if line.startswith("#"):
                if line.startswith("# hipe2022:document_id"):
                    flush_document()  # close previous doc if metadata restarts without blank line
                    doc_id = line.split("=", 1)[1].strip()
                    current_doc = Document(doc_id=doc_id)
                elif "=" in line and current_doc is not None:
                    key, _, val = line[2:].partition("=")
                    current_doc.metadata[key.strip()] = val.strip()
                continue

            if line.startswith("TOKEN"):
                continue  # header line

            cols = line.split("\t")
            if len(cols) < 2:
                continue
            token, ne_coarse_lit = cols[0], cols[1]
            misc = cols[-1] if len(cols) >= 10 else ""

            if current_doc is None:
                # Defensive: shouldn't happen given well-formed HIPE files
                current_doc = Document(doc_id="unknown")

            current_tokens.append(token)
            current_labels.append(ne_coarse_lit if ne_coarse_lit != "_" else "O")

            if "EndOfSentence" in misc:
                flush_sentence()

    flush_document()
    return documents


def to_sentence_examples(documents: list[Document]) -> list[dict]:
    """Flatten documents into a list of {tokens, ner_tags, doc_id} examples,
    one per sentence."""
    examples = []
    for doc in documents:
        for tokens, labels in doc.sentences:
            examples.append({"tokens": tokens, "ner_tags": labels, "doc_id": doc.doc_id})
    return examples


def collect_label_set(examples: list[dict]) -> list[str]:
    labels = set()
    for ex in examples:
        labels.update(ex["ner_tags"])
    # Ensure a stable, IOB-conventional ordering with O first
    ordered = ["O"] + sorted(l for l in labels if l != "O")
    return ordered


if __name__ == "__main__":
    import sys
    docs = parse_hipe_tsv(sys.argv[1] if len(sys.argv) > 1 else
                           "data/topres19th_raw/en/HIPE-2022-v2.1-topres19th-train-en.tsv")
    examples = to_sentence_examples(docs)
    print(f"Parsed {len(docs)} documents, {len(examples)} sentences")
    print(f"Label set: {collect_label_set(examples)}")
    print(f"Example: {examples[0]}")
