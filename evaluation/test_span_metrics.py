"""Hand-verified unit tests for the span-based metric, ensuring it computes
exact-span-match precision/recall/F1 correctly before trusting it for real
evaluation (same discipline as the TFN fusion layer's unit test)."""
from evaluation.span_metrics import extract_spans, compute_span_prf


def test_extract_spans_basic():
    labels = ["O", "B-LOC", "I-LOC", "O", "B-STREET", "O"]
    spans = extract_spans(labels)
    assert spans == {("LOC", 1, 3), ("STREET", 4, 5)}, spans


def test_extract_spans_adjacent_entities():
    # two B- tags back to back with no O between: two separate spans
    labels = ["B-LOC", "B-BUILDING"]
    spans = extract_spans(labels)
    assert spans == {("LOC", 0, 1), ("BUILDING", 1, 2)}, spans


def test_extract_spans_malformed_i_without_b():
    # I-LOC with no preceding B-LOC: tolerated as its own span (models can
    # produce this during early training)
    labels = ["O", "I-LOC", "O"]
    spans = extract_spans(labels)
    assert spans == {("LOC", 1, 2)}, spans


def test_perfect_prediction():
    true = [["O", "B-LOC", "I-LOC", "O"]]
    pred = [["O", "B-LOC", "I-LOC", "O"]]
    result = compute_span_prf(true, pred)
    assert result["overall_precision"] == 1.0
    assert result["overall_recall"] == 1.0
    assert result["overall_f1"] == 1.0


def test_completely_wrong_prediction():
    true = [["O", "B-LOC", "I-LOC", "O"]]
    pred = [["O", "O", "O", "O"]]
    result = compute_span_prf(true, pred)
    assert result["overall_precision"] == 0.0  # no predicted spans -> precision undefined, treated as 0
    assert result["overall_recall"] == 0.0
    assert result["overall_f1"] == 0.0


def test_partial_overlap_counts_as_wrong():
    # exact span match required: predicting B-LOC I-LOC I-LOC when true is
    # B-LOC I-LOC (one token too long) should NOT count as correct
    true = [["B-LOC", "I-LOC", "O"]]
    pred = [["B-LOC", "I-LOC", "I-LOC"]]
    # true span: (LOC, 0, 2). pred span: (LOC, 0, 3) -- different span, no match
    result = compute_span_prf(true, pred)
    assert result["overall_precision"] == 0.0
    assert result["overall_recall"] == 0.0


def test_per_type_breakdown():
    true = [["B-LOC", "O", "B-STREET"]]
    pred = [["B-LOC", "O", "O"]]  # got LOC right, missed STREET entirely
    result = compute_span_prf(true, pred)
    assert result["per_type"]["LOC"]["f1"] == 1.0
    assert result["per_type"]["STREET"]["f1"] == 0.0
    assert result["per_type"]["STREET"]["support"] == 1


def test_hand_computed_precision_recall():
    # 2 true spans, model predicts 3 spans, 1 of which is correct
    true = [["B-LOC", "O", "B-LOC", "O"]]
    pred = [["B-LOC", "O", "O", "B-LOC"]]  # first LOC correct, second LOC wrong position
    result = compute_span_prf(true, pred)
    # true spans: (LOC,0,1), (LOC,2,3). pred spans: (LOC,0,1), (LOC,3,4)
    # tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5, f1=0.5
    assert abs(result["overall_precision"] - 0.5) < 1e-9
    assert abs(result["overall_recall"] - 0.5) < 1e-9
    assert abs(result["overall_f1"] - 0.5) < 1e-9


if __name__ == "__main__":
    test_extract_spans_basic()
    test_extract_spans_adjacent_entities()
    test_extract_spans_malformed_i_without_b()
    test_perfect_prediction()
    test_completely_wrong_prediction()
    test_partial_overlap_counts_as_wrong()
    test_per_type_breakdown()
    test_hand_computed_precision_recall()
    print("All span_metrics tests passed.")
