import unittest

from src.metrics import (
    accuracy,
    confusion_counts,
    evaluate,
    f1_score,
    load_predictions,
    precision,
    recall,
)


class TestMetrics(unittest.TestCase):
    DATA_PATH = "data/predictions.csv"

    def setUp(self):
        self.y_true, self.y_pred = load_predictions(self.DATA_PATH)

    def test_confusion_counts(self):
        self.assertEqual(confusion_counts(self.y_true, self.y_pred), (5, 1, 2, 4))

    def test_precision(self):
        self.assertAlmostEqual(precision(self.y_true, self.y_pred), 5 / 6, places=6)

    def test_recall(self):
        self.assertAlmostEqual(recall(self.y_true, self.y_pred), 5 / 7, places=6)

    def test_f1_score(self):
        self.assertAlmostEqual(f1_score(self.y_true, self.y_pred), 10 / 13, places=6)

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy(self.y_true, self.y_pred), 0.75, places=6)

    def test_zero_division_is_handled(self):
        self.assertEqual(precision([0, 0], [0, 0]), 0.0)
        self.assertEqual(recall([0, 0], [0, 0]), 0.0)
        self.assertEqual(f1_score([0, 0], [0, 0]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            confusion_counts([1, 0, 1], [1, 0])

    def test_evaluate_keys(self):
        self.assertEqual(
            set(evaluate(self.DATA_PATH)),
            {"precision", "recall", "f1", "accuracy"},
        )


if __name__ == "__main__":
    unittest.main()
