"""Fine-tunes a small transformer for NER on the topres19th (historical
English newspaper place names) dataset. Uses HuggingFace transformers'
Trainer, with our own span-based metric (evaluation/span_metrics.py)
instead of seqeval (which fails to install in this environment).
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from data.hipe_parser import parse_hipe_tsv, to_sentence_examples, collect_label_set
from data.prepare_dataset import build_label_maps, examples_to_hf_dataset, make_tokenize_and_align_fn
from evaluation.span_metrics import compute_span_prf


def load_split(data_dir: str, split: str):
    path = os.path.join(data_dir, f"HIPE-2022-v2.1-topres19th-{split}-en.tsv")
    docs = parse_hipe_tsv(path)
    return to_sentence_examples(docs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/topres19th_raw/en")
    parser.add_argument("--model_name", default="distilbert-base-cased")
    parser.add_argument("--output_dir", default="results/model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_train_examples", type=int, default=None,
                         help="subsample for quick pipeline checks")
    args = parser.parse_args()

    print("Loading and parsing HIPE data...")
    train_examples = load_split(args.data_dir, "train")
    dev_examples = load_split(args.data_dir, "dev")
    test_examples = load_split(args.data_dir, "test")

    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]

    label_list = collect_label_set(train_examples + dev_examples + test_examples)
    label2id, id2label = build_label_maps(label_list)
    print(f"Labels: {label_list}")
    print(f"Train/dev/test sentences: {len(train_examples)}/{len(dev_examples)}/{len(test_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenize_fn = make_tokenize_and_align_fn(tokenizer, max_length=64)

    train_ds = examples_to_hf_dataset(train_examples, label2id).map(
        tokenize_fn, batched=True, remove_columns=["tokens", "ner_tags", "doc_id"]
    )
    dev_ds = examples_to_hf_dataset(dev_examples, label2id).map(
        tokenize_fn, batched=True, remove_columns=["tokens", "ner_tags", "doc_id"]
    )
    test_ds = examples_to_hf_dataset(test_examples, label2id).map(
        tokenize_fn, batched=True, remove_columns=["tokens", "ner_tags", "doc_id"]
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_labels_list, pred_labels_list = [], []
        for pred_row, label_row in zip(predictions, labels):
            true_labels, pred_labels = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_labels.append(id2label[l])
                pred_labels.append(id2label[p])
            true_labels_list.append(true_labels)
            pred_labels_list.append(pred_labels)
        result = compute_span_prf(true_labels_list, pred_labels_list)
        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"],
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("\nFinal evaluation on TEST set:")
    test_results = trainer.evaluate(test_ds)
    print(test_results)

    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    return test_results


if __name__ == "__main__":
    main()
