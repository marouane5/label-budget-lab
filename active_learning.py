"""Pool-based active learning for multiclass classification using NumPy."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Checkpoint:
    budget: int
    acquired: np.ndarray
    weights: np.ndarray


def add_intercept(features):
    return np.column_stack((features, np.ones(len(features))))


def softmax(scores):
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_classifier(features, targets, n_classes, iterations=180, rate=0.4, penalty=0.002):
    """Fit softmax regression from zero at each budget, with L2 on pixel weights."""
    design = add_intercept(features)
    encoded = np.eye(n_classes)[targets]
    weights = np.zeros((design.shape[1], n_classes))
    for _ in range(iterations):
        residual = softmax(design @ weights) - encoded
        gradient = design.T @ residual / len(targets)
        gradient[:-1] += penalty * weights[:-1]
        weights -= rate * gradient
    return weights


def entropy(probabilities):
    # x log(x) tends to zero at zero.
    tiny = np.finfo(float).tiny
    return -(probabilities * np.log(np.maximum(probabilities, tiny))).sum(axis=1)


def choose_batch(strategy, available, count, pool_features, weights, rng):
    if strategy == "random":
        return rng.choice(available, size=count, replace=False)
    if strategy == "entropy":
        probabilities = softmax(add_intercept(pool_features[available]) @ weights)
        # available is ordered by row ID, so stable sorting also fixes exact ties.
        order = np.argsort(-entropy(probabilities), kind="stable")
        return available[order[:count]]
    raise ValueError(f"Unknown strategy: {strategy}")


def learn(pool_features, reveal_labels, initial, budgets, strategy, rng, n_classes=10):
    """Access training labels only after selecting a batch. No evaluation data enters here."""
    budgets = tuple(budgets)
    initial = np.asarray(initial, dtype=int)
    if (not budgets or len(initial) != budgets[0] or budgets[0] < 1
            or any(b <= a for a, b in zip(budgets, budgets[1:]))
            or budgets[-1] > len(pool_features)):
        raise ValueError("Budgets must grow from the initial sample size to at most the pool size.")
    if len(np.unique(initial)) != len(initial) or np.any((initial < 0) | (initial >= len(pool_features))):
        raise ValueError("Initial row IDs must be unique and inside the pool.")
    available = np.ones(len(pool_features), dtype=bool)
    acquired = initial
    all_ids = []
    revealed = []
    checkpoints = []
    for budget in budgets:
        if all_ids:
            count = budget - len(all_ids)
            acquired = choose_batch(strategy, np.flatnonzero(available), count, pool_features, weights, rng)
        available[acquired] = False
        all_ids.extend(acquired.tolist())
        revealed.extend(reveal_labels(acquired).tolist())
        weights = fit_classifier(pool_features[all_ids], np.asarray(revealed), n_classes)
        checkpoints.append(Checkpoint(budget, acquired.copy(), weights))
    return checkpoints


def classification_metrics(actual, predicted, n_classes=10):
    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (actual, predicted), 1)
    diagonal = np.diag(counts)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(n_classes), where=denominator > 0)
    return float(diagonal.sum() / len(actual)), float(f1.mean())
