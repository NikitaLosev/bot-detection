"""Тесты метрики, временных разбиений и границ окна"""

import unittest

import numpy as np
import pandas as pd

import scripts

evaluate = scripts.evaluate_predictions


def by_definition(y_true, score):
    """Считает метрику перебором порогов: максимум precision при recall не ниже 0.70"""
    points = []
    for threshold in np.unique(score):
        flagged = score >= threshold
        caught = (flagged & (y_true == 1)).sum()
        points.append((caught / flagged.sum(), caught / y_true.sum()))
    allowed = [(precision, recall) for precision, recall in points if recall >= 0.7]
    best = max(precision for precision, _ in allowed)
    return best, max(recall for precision, recall in allowed if precision == best)


class TestMetric(unittest.TestCase):
    """Максимум precision среди допустимых порогов и неделимые группы равных score"""

    def test_takes_best_allowed_point(self):
        """Первая точка с recall не ниже 0.70 даёт 0.75, следующая 0.8, ответ 0.8"""
        metrics = evaluate([1, 0, 1, 1, 1], [0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertEqual(metrics['precision_at_recall_070'], 0.8)
        self.assertEqual(metrics['official_point_recall_on_eval'], 1.0)
        # При равной precision в нескольких точках берётся наибольший recall
        several = evaluate([0, 0, 0] + [1] * 9 + [0, 1, 1, 1], np.linspace(0.95, 0.2, 16))
        self.assertEqual(several['precision_at_recall_070'], 0.75)
        self.assertEqual(several['official_point_recall_on_eval'], 1.0)

    def test_ties_and_row_order(self):
        """Группа равных score отмечается целиком, перестановка строк ничего не меняет"""
        metrics = evaluate([1, 1, 0, 0], [0.5] * 4)
        self.assertEqual(metrics['precision_at_recall_070'], 0.5)
        self.assertEqual(metrics['tied_score_share'], 1.0)
        self.assertEqual(scripts.tied_score_share(np.array([0.8, 0.8, 0.2])), 2 / 3)
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        score = np.array([0.9, 0.9, 0.7, 0.4, 0.4, 0.4, 0.2, 0.1])
        order = [5, 2, 7, 0, 3, 6, 1, 4]
        base = evaluate(y_true, score)
        self.assertEqual(base['tied_score_share'], 5 / 8)
        shuffled = evaluate(y_true[order], score[order])
        self.assertTrue(np.allclose(list(base.values()), list(shuffled.values()), equal_nan=True))

    def test_matches_definition(self):
        """На случайных выборках с равными score совпадает с перебором порогов"""
        generator = np.random.default_rng(0)
        for _ in range(50):
            y_true = generator.integers(0, 2, 40)
            y_true[0] = 1
            score = generator.integers(0, 9, 40) / 8
            metrics = evaluate(y_true, score)
            best, recall = by_definition(y_true, score)
            self.assertAlmostEqual(metrics['precision_at_recall_070'], best)
            self.assertAlmostEqual(metrics['official_point_recall_on_eval'], recall)

    def test_rejects_bad_input(self):
        """Некорректные метки и score вызывают ошибку"""
        bad_inputs = [
            ([], []), ([1, 0], [0.5]), ([[1, 0]], [[0.5, 0.2]]), ([1, 2], [0.5, 0.2]),
            ([1, np.nan], [0.5, 0.2]), ([1, 0], [0.5, np.nan]), ([1, 0], [0.5, np.inf]),
            ([1, 0], [0.5, 1.2]), ([1, 0], [0.5, -0.1]),
        ]
        for y_true, score in bad_inputs:
            with self.assertRaises(ValueError):
                evaluate(y_true, score)


if __name__ == '__main__':
    unittest.main()
