"""Converts HIPE sentence examples into a tokenized HuggingFace Dataset,
handling the subword-alignment problem: the tokenizer splits words into
subword pieces, but our labels are per-word, so we must propagate each
word's label to only its first subword token (standard NER fine-tuning
practice) and mask the rest with -100 (ignored by the loss).
"""
from __future__ import annotations

from datasets import Dataset


def build_label_maps(label_list: list[str]) -> tuple[dict, dict]:
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def examples_to_hf_dataset(examples: list[dict], label2id: dict) -> Dataset:
    return Dataset.from_dict({
        "tokens": [ex["tokens"] for ex in examples],
        "ner_tags": [[label2id[l] for l in ex["ner_tags"]] for ex in examples],
        "doc_id": [ex["doc_id"] for ex in examples],
    })


def make_tokenize_and_align_fn(tokenizer, label_all_subword_tokens: bool = False, max_length: int = 64):
    """label_all_subword_tokens=False (default, standard practice): only the
    first subword of each word gets the real label; subsequent subwords of
    the same word get -100 (ignored in loss computation)."""

    def tokenize_and_align_labels(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )
        all_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                else:
                    label_ids.append(labels[word_idx] if label_all_subword_tokens else -100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    return tokenize_and_align_labels
