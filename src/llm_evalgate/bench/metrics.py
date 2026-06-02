from __future__ import annotations


def _validate(predicted: list[bool], labels: list[bool]) -> None:
    if len(predicted) != len(labels):
        raise ValueError(
            f"predicted and labels must be equal length; "
            f"got {len(predicted)} and {len(labels)}"
        )
    if not predicted:
        raise ValueError("predicted and labels must be non-empty")


def confusion_counts(predicted: list[bool], labels: list[bool]) -> dict[str, int]:
    """Return tp/fp/tn/fn counts with the positive class being ``True`` (pass)."""
    _validate(predicted, labels)
    tp = fp = tn = fn = 0
    for pred, label in zip(predicted, labels):
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and not label:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def accuracy(predicted: list[bool], labels: list[bool]) -> float:
    _validate(predicted, labels)
    correct = sum(1 for pred, label in zip(predicted, labels) if pred == label)
    return correct / len(labels)


def precision(predicted: list[bool], labels: list[bool]) -> float:
    """Precision for the positive class (pass). Zero denominator returns 0.0."""
    counts = confusion_counts(predicted, labels)
    denom = counts["tp"] + counts["fp"]
    if denom == 0:
        return 0.0
    return counts["tp"] / denom


def recall(predicted: list[bool], labels: list[bool]) -> float:
    """Recall for the positive class (pass). Zero denominator returns 0.0."""
    counts = confusion_counts(predicted, labels)
    denom = counts["tp"] + counts["fn"]
    if denom == 0:
        return 0.0
    return counts["tp"] / denom


def f1(predicted: list[bool], labels: list[bool]) -> float:
    """Harmonic mean of precision and recall. Zero denominator returns 0.0."""
    prec = precision(predicted, labels)
    rec = recall(predicted, labels)
    denom = prec + rec
    if denom == 0:
        return 0.0
    return 2 * prec * rec / denom


def cohen_kappa(predicted: list[bool], labels: list[bool]) -> float:
    """Cohen's kappa for two binary raters, corrected for chance agreement.

    Zero expected-disagreement denominator returns 0.0.
    """
    counts = confusion_counts(predicted, labels)
    n = len(labels)
    observed = (counts["tp"] + counts["tn"]) / n
    pred_pos = (counts["tp"] + counts["fp"]) / n
    label_pos = (counts["tp"] + counts["fn"]) / n
    expected = pred_pos * label_pos + (1 - pred_pos) * (1 - label_pos)
    denom = 1 - expected
    if denom == 0:
        return 0.0
    return (observed - expected) / denom


def regression_catch_rate(predicted: list[bool], labels: list[bool]) -> float:
    """Fraction of regressions (label ``False``) correctly predicted ``False``.

    This is recall on the negative class. If there are no regressions in
    ``labels`` the denominator is zero and this returns 0.0; it is the
    caller's job to note that no regressions were present.
    """
    counts = confusion_counts(predicted, labels)
    denom = counts["tn"] + counts["fp"]
    if denom == 0:
        return 0.0
    return counts["tn"] / denom


def all_metrics(predicted: list[bool], labels: list[bool]) -> dict[str, float]:
    """Compute the full metric suite in one pass-style call."""
    return {
        "accuracy": accuracy(predicted, labels),
        "precision": precision(predicted, labels),
        "recall": recall(predicted, labels),
        "f1": f1(predicted, labels),
        "cohen_kappa": cohen_kappa(predicted, labels),
        "regression_catch_rate": regression_catch_rate(predicted, labels),
    }
