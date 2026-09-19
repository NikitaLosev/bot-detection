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


def toy_meta():
    """Три cookie в день с 04-06 по 04-20, части как в данных, бот первый за день"""
    starts = pd.date_range('2026-04-06', '2026-04-20').repeat(3)
    meta = pd.DataFrame({
        'cookie_id': [f'c{number:02d}' for number in range(len(starts))],
        'window_start_ts': starts, 'window_end_ts': starts + pd.Timedelta(days=1),
        'cookie_created_at': starts - pd.Timedelta(hours=5),
    })
    meta['part'] = np.select(
        [starts >= pd.Timestamp('2026-04-20'), starts >= scripts.HOLDOUT_START],
        ['test', 'holdout'], 'research')
    is_bot = (np.arange(len(meta)) % 3 == 0).astype(float)
    meta['target'] = np.where(meta['part'] == 'research', is_bot, np.nan)
    return meta


def toy_events(rows):
    """События из кортежей (cookie_id, время, тип) со всеми исходными колонками"""
    events = pd.DataFrame({column: np.nan for column in scripts.RAW_EVENT_COLUMNS},
                          index=range(len(rows)))
    cookies, times, names = zip(*rows)
    return events.assign(cookie_id=list(cookies), event_ts=pd.to_datetime(list(times)),
                         eid=range(len(rows)), event_name=list(names), platform='WEB',
                         user_agent='Mozilla/5.0', search_page=1.0, row_in_file=range(len(rows)))


def toy_predictions(folds):
    """Случайные score для проверки сопоставления, к качеству модели отношения не имеют"""
    generator = np.random.default_rng(0)
    parts = [pd.DataFrame({'fold': fold, 'cookie_id': parts['valid'],
                           'score': generator.random(len(parts['valid']))})
             for fold, parts in folds.items()]
    return pd.concat(parts, ignore_index=True)


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


class TestValidation(unittest.TestCase):
    """Временные разбиения и сопоставление предсказаний с метками по cookie_id"""

    def test_folds_take_whole_days(self):
        """Разбиения берут целые дни, обучение раньше проверки, порядок строк не влияет"""
        meta = toy_meta()
        folds = scripts.fold_cookies(meta)
        day_of = meta.set_index('cookie_id')['window_start_ts'].dt.strftime('%m-%d')
        expected_days = {
            ('fold_a', 'train'): ['04-06', '04-07', '04-08', '04-09'],
            ('fold_a', 'valid'): ['04-10', '04-11', '04-12'],
            ('fold_b', 'train'): ['04-06', '04-07', '04-08', '04-09', '04-10', '04-11', '04-12'],
            ('fold_b', 'valid'): ['04-13', '04-14', '04-15', '04-16'],
        }
        for (fold, role), days in expected_days.items():
            ids = folds[fold][role]
            self.assertEqual(sorted(day_of[ids].unique()), days)
            self.assertEqual(len(ids), 3 * len(days))
        for parts in folds.values():
            self.assertEqual(len(parts['train'].intersection(parts['valid'])), 0)
        shuffled = scripts.fold_cookies(meta.sample(frac=1, random_state=1))
        for fold, parts in shuffled.items():
            self.assertTrue(parts['train'].equals(folds[fold]['train']))
            self.assertTrue(parts['valid'].equals(folds[fold]['valid']))

    def test_fold_guard_rejects_bad_parts(self):
        """Пересечение, отложенный блок в разбиении и обучение позже проверки отклоняются"""
        meta = toy_meta()
        ids = meta.set_index('window_start_ts')['cookie_id']
        early, late = pd.Index(ids['2026-04-06']), pd.Index(ids['2026-04-10'])
        holdout = pd.Index(ids['2026-04-17'])
        for train, valid in [(early, early.union(late)), (early, late.union(holdout)),
                             (late, early)]:
            with self.assertRaises(ValueError):
                scripts.check_fold(meta, train, valid)

    def test_evaluate_folds_averages_two_splits(self):
        """Среднее равно среднему двух метрик, порядок строк и чужой target не влияют"""
        meta = toy_meta()
        folds = scripts.fold_cookies(meta)
        predictions = toy_predictions(folds)
        table, mean = scripts.evaluate_folds(predictions, meta, folds)
        labels = meta.set_index('cookie_id')['target']
        official = [scripts.precision_at_recall(labels[part['cookie_id']], part['score'],
                                                recall=scripts.TARGET_RECALL)
                    for _, part in predictions.groupby('fold')]
        self.assertAlmostEqual(mean, float(np.mean(official)))
        self.assertEqual(table['n_cookies'].tolist(), [9, 12])
        other, _ = scripts.evaluate_folds(
            predictions.sample(frac=1, random_state=2).assign(target=1), meta, folds)
        self.assertTrue(other.equals(table))

    def test_evaluate_folds_rejects_wrong_cookies(self):
        """Пропущенная, лишняя, повторная и чужая cookie и неполный набор разбиений отклоняются"""
        meta = toy_meta()
        folds = scripts.fold_cookies(meta)
        predictions = toy_predictions(folds)
        train_only = folds['fold_a']['train'][0]
        extra = pd.DataFrame({'fold': ['fold_a'], 'cookie_id': [train_only], 'score': [0.5]})
        swapped = predictions.copy()
        swapped.loc[0, 'cookie_id'] = train_only
        bad = [predictions.drop(index=0), pd.concat([predictions, extra]),
               pd.concat([predictions, predictions.head(1)]), swapped,
               predictions[predictions['fold'] == 'fold_a']]
        for wrong in bad:
            with self.assertRaises(ValueError):
                scripts.evaluate_folds(wrong, meta, folds)
        with self.assertRaises(ValueError):
            scripts.evaluate_folds(predictions, meta, {'fold_b': folds['fold_b']})


class TestWindow(unittest.TestCase):
    """Границы окна и cookie без событий внутри него"""

    def test_window_borders_and_empty_cookie(self):
        """Начало окна входит, конец исключается, cookie без событий остаётся с нулями"""
        meta = toy_meta().head(2)
        start = meta.loc[0, 'window_start_ts']
        rows = [('c00', start - pd.Timedelta(seconds=1), 'item_view'),
                ('c00', start, 'item_view'),
                ('c00', start + pd.Timedelta(hours=10), 'search_results_view'),
                ('c00', start + pd.Timedelta(days=1), 'item_view'),
                ('c01', start + pd.Timedelta(days=1, hours=1), 'item_view')]
        clean = scripts.clean_events(scripts.window_events(toy_events(rows), meta))
        self.assertEqual(len(clean), 2)
        features = scripts.cookie_features(clean, meta)
        self.assertEqual(list(features.index), ['c00', 'c01'])
        self.assertEqual(features.loc['c00', 'n_events'], 2)
        self.assertEqual(features.loc['c00', 'gap_median_s'], 10 * 3600)
        self.assertEqual(features.loc['c01', 'n_events'], 0)
        self.assertTrue(np.isnan(features.loc['c01', 'gap_median_s']))


if __name__ == '__main__':
    unittest.main()
