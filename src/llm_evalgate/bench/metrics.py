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


def _validate_pair(x: list[float], y: list[float]) -> None:
    if len(x) != len(y):
        raise ValueError(
            f"x and y must be equal length; got {len(x)} and {len(y)}"
        )
    if not x:
        raise ValueError("x and y must be non-empty")


def pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient. Zero-variance input returns 0.0."""
    _validate_pair(x, y)
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [xi - mean_x for xi in x]
    dy = [yi - mean_y for yi in y]
    covariance = sum(a * b for a, b in zip(dx, dy))
    var_x = sum(a * a for a in dx)
    var_y = sum(b * b for b in dy)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return 0.0
    return covariance / denom


def _average_ranks(values: list[float]) -> list[float]:
    """Rank ``values`` ascending, assigning tied values their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation: Pearson on the average-rank transform."""
    _validate_pair(x, y)
    return pearson(_average_ranks(x), _average_ranks(y))


def mae(x: list[float], y: list[float]) -> float:
    """Mean absolute error between two equal-length vectors."""
    _validate_pair(x, y)
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x)
