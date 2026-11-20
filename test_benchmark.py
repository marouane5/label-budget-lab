import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from active_learning import add_intercept, choose_batch, classification_metrics, entropy, fit_classifier, learn, softmax
from benchmark import run_experiment, summarize
from digits_data import read_digits


class AcquisitionTests(unittest.TestCase):
    def test_uniform_distribution_has_maximum_entropy(self):
        scores = np.array([[1000.0, 1000.0], [1000.0, -1000.0]])
        probabilities = softmax(scores)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(entropy(probabilities), [np.log(2), 0.0])

    def test_entropy_selects_ambiguous_point_and_breaks_ties_by_input_order(self):
        features = np.array([[5.0], [0.0], [0.0], [-6.0]])
        weights = np.array([[1.0, -1.0], [0.0, 0.0]])
        selection = choose_batch("entropy", np.arange(4), 2, features, weights, np.random.default_rng(9))
        np.testing.assert_array_equal(selection, [1, 2])

    def test_fit_separates_two_classes(self):
        features = np.array([[-3.0], [-1.0], [1.0], [3.0]])
        labels = np.array([0, 0, 1, 1])
        prediction = (add_intercept(features) @ fit_classifier(features, labels, 2)).argmax(axis=1)
        np.testing.assert_array_equal(prediction, labels)

    def test_only_queried_labels_are_requested_once(self):
        features = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        for strategy in ("entropy", "random"):
            calls = []

            def oracle(ids):
                calls.extend(ids.tolist())
                return labels[ids]

            trace = learn(features, oracle, [0, 4], [2, 4, 5], strategy, np.random.default_rng(3), n_classes=2)
            self.assertEqual([len(point.acquired) for point in trace], [2, 2, 1])
            self.assertEqual(calls, np.concatenate([point.acquired for point in trace]).tolist())
            self.assertEqual(len(set(calls)), 5)

    def test_bad_initial_ids_and_budgets_raise(self):
        pixels = np.ones((4, 2))
        for initial, budgets in [([0, 0], [2, 3]), ([4], [1, 2]), ([0], [1, 1]), ([0], [1, 5])]:
            with self.subTest(initial=initial, budgets=budgets), self.assertRaises(ValueError):
                learn(pixels, None, initial, budgets, "random", np.random.default_rng(4))


class EvaluationTests(unittest.TestCase):
    def test_known_macro_f1_and_accuracy(self):
        accuracy, f1 = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], n_classes=2)
        self.assertAlmostEqual(accuracy, 0.75)
        self.assertAlmostEqual(f1, (2 / 3 + 4 / 5) / 2)

    def test_test_population_cannot_change_acquisitions(self):
        rng = np.random.default_rng(7)
        train = (rng.normal(size=(30, 4)), np.tile(np.arange(10), 3))
        test = (rng.normal(size=(10, 4)), np.arange(10))
        scores, queries = run_experiment(train, test, [19], [4, 9])
        changed_test = (-test[0], test[1][::-1])
        _, changed_queries = run_experiment(train, changed_test, [19], [4, 9])
        self.assertEqual(queries, changed_queries)
        self.assertEqual(scores[0]["accuracy"], scores[2]["accuracy"])
        initial_random = [row["train_row"] for row in queries if row["strategy"] == "random" and row["labels"] == 4]
        initial_entropy = [row["train_row"] for row in queries if row["strategy"] == "entropy" and row["labels"] == 4]
        self.assertEqual(initial_random, initial_entropy)
        self.assertIsNone(summarize(scores, [19], [4, 9])[-1]["paired_accuracy_gain_sd"])

    def test_summary_pairs_within_each_seed(self):
        rows = [
            dict(seed=s, strategy=method, labels=6, accuracy=value, macro_f1=value)
            for s, method, value in [(1, "random", .5), (2, "entropy", .7), (1, "entropy", .4), (2, "random", .4)]
        ]
        summary = summarize(rows, [1, 2], [6])[0]
        self.assertAlmostEqual(summary["paired_accuracy_gain"], .1)
        self.assertEqual(summary["entropy_accuracy_wins"], 1)

    def test_changed_dataset_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("optdigits.tra", "wrong content")
            with self.assertRaisesRegex(ValueError, "Unexpected contents"):
                read_digits(path)


if __name__ == "__main__":
    unittest.main()
