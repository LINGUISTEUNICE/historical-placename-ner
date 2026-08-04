"""Span-based precision/recall/F1 for IOB-tagged sequences, implemented
directly (not via seqeval, which fails to build in this environment --
see research_diary.md). This computes the same standard metric seqeval
would (exact span match, not per-token accuracy), just without the
dependency. Validated against hand-constructed cases in
evaluation/test_metrics.py before being trusted for real evaluation.
"""
from __future__ import annotations

from collections import defaultdict


def extract_spans(labels: list[str]) -> set[tuple]:
    """Extract (entity_type, start_idx, end_idx) spans from an IOB label
    sequence. end_idx is exclusive. A malformed sequence (e.g. I-LOC with no
    preceding B-LOC) is tolerated by treating a lone I- tag as starting a
    new span -- common practice, since model predictions are often
    imperfect and we still want a sensible span extraction."""
    spans = set()
    start = None
    current_type = None
    for i, label in enumerate(labels + ["O"]):  # sentinel to flush last span
        if label == "O":
            if start is not None:
                spans.add((current_type, start, i))
            start = None
            current_type = None
        elif label.startswith("B-"):
            if start is not None:
                spans.add((current_type, start, i))
            start = i
            current_type = label[2:]
        elif label.startswith("I-"):
            ent_type = label[2:]
            if start is None or ent_type != current_type:
                # malformed: I- without matching B-, start a new span here
                if start is not None:
                    spans.add((current_type, start, i))
                start = i
                current_type = ent_type
            # else: continuation of current span, nothing to do
    return spans


def compute_span_prf(
    true_labels_list: list[list[str]], pred_labels_list: list[list[str]]
) -> dict:
    """Returns overall micro-averaged precision/recall/F1, plus per-entity-type
    breakdown, computed over exact span matches (type + boundaries)."""
    assert len(true_labels_list) == len(pred_labels_list)

    tp_by_type = defaultdict(int)
    fp_by_type = defaultdict(int)
    fn_by_type = defaultdict(int)

    for true_labels, pred_labels in zip(true_labels_list, pred_labels_list):
        true_spans = extract_spans(true_labels)
        pred_spans = extract_spans(pred_labels)

        for span in pred_spans:
            ent_type = span[0]
            if span in true_spans:
                tp_by_type[ent_type] += 1
            else:
                fp_by_type[ent_type] += 1
        for span in true_spans:
            if span not in pred_spans:
                fn_by_type[span[0]] += 1

    all_types = set(tp_by_type) | set(fp_by_type) | set(fn_by_type)
    per_type = {}
    total_tp = total_fp = total_fn = 0
    for t in sorted(all_types):
        tp, fp, fn = tp_by_type[t], fp_by_type[t], fn_by_type[t]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_type[t] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    return {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "per_type": per_type,
    }
