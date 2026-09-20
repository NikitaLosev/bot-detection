"""Данные, признаки, метрика и временные разбиения"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)

# Исходная метрика задачи
from bot_detection_challenge.metric import (
    TARGET_RECALL, pr_curve, precision_at_recall, recall_at_fpr,
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / 'bot_detection_challenge' / 'data'
META_DATES = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
# Последние три дня train закрыты при исследовании, обоснование в eda.ipynb
HOLDOUT_START = pd.Timestamp('2026-04-17')
# desktop и web, iphone и ios обозначают одну платформу
PLATFORM_MAP = {'desktop': 'web', 'iphone': 'ios'}
RAW_EVENT_COLUMNS = [
    'cookie_id', 'event_ts', 'eid', 'event_name', 'platform', 'user_agent',
    'item_id', 'item_category', 'item_location', 'seller_type',
    'search_query', 'search_page', 'pointer_x', 'pointer_y',
]
# Словари заданы кодом нормализации, а не данными, поэтому не зависят от разбиения
PLATFORMS = ['web', 'android', 'ios']
UA_KINDS = ['browser', 'app', 'headless', 'http_lib']
EVENT_TYPES = [
    'item_view', 'search_results_view', 'photo_swipe', 'favorite_add', 'seller_page_view',
    'contact_phone_show', 'contact_chat_open', 'contact_message_sent', 'login',
]
# Наблюдаемые границы координат курсора, значения целые
SCREEN_W, SCREEN_H = 1920, 1080
# Внутренние разбиения: дни начала окна, обе даты включительно
FOLDS = {
    'fold_a': {'train': ('2026-04-06', '2026-04-09'), 'valid': ('2026-04-10', '2026-04-12')},
    'fold_b': {'train': ('2026-04-06', '2026-04-12'), 'valid': ('2026-04-13', '2026-04-16')},
}
# Диагностическая точка с условным порогом, recall 0.70 в ней не требуется
DIAGNOSTIC_THRESHOLD = 0.5
MAX_FALSE_POSITIVE_RATE = 0.01
# Финальная модель: среднее вероятностей трёх CatBoost на признаках model_features
FINAL_PARAMS = {'iterations': 1500, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 10}
FINAL_SEEDS = (0, 17, 42)
THREADS = 4


def read_meta(open_holdout=False):
    """Читает train и test в одну таблицу cookie, по умолчанию метки 04-17..04-19 скрыты"""
    train = pd.read_csv(DATA_DIR / 'train.csv', parse_dates=META_DATES)
    test = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=META_DATES)
    is_holdout = train['window_start_ts'] >= HOLDOUT_START
    train['part'] = np.where(is_holdout, 'holdout', 'research')
    test['part'] = 'test'
    meta = pd.concat([train, test], ignore_index=True)
    if not open_holdout:
        # NaN вместо метки: случайный расчёт по блоку сразу даст пропуск, а не число
        meta['target'] = meta['target'].where(meta['part'] != 'holdout')
    return meta


def read_events():
    """Читает события и запоминает номер строки в исходном файле"""
    events = pd.read_csv(DATA_DIR / 'events.csv.gz', parse_dates=['event_ts'])
    events['row_in_file'] = np.arange(len(events))
    return events


def attach_window(events, meta):
    """Присоединяет к событиям окно их cookie и отмечает положение события относительно окна"""
    keys = meta[['cookie_id', 'window_start_ts', 'window_end_ts', 'part']]
    merged = events.merge(keys, on='cookie_id', how='left', validate='many_to_one')
    assert len(merged) == len(events), 'соединение изменило число событий'
    assert merged['window_start_ts'].notna().all(), 'у события нет cookie в train или test'
    ts, start, end = merged['event_ts'], merged['window_start_ts'], merged['window_end_ts']
    conditions = [ts < start, ts < end, ts == end]
    merged['position'] = np.select(conditions, ['before', 'inside', 'at_end'], 'after')
    return merged


def window_events(events, meta):
    """Оставляет события внутри окна: window_start_ts <= event_ts < window_end_ts"""
    merged = attach_window(events, meta)
    return merged[merged['position'] == 'inside'].drop(columns='position')


def normalize_platform(platform):
    """Сводит 11 написаний платформы к трём значениям: web, android, ios"""
    return platform.str.lower().replace(PLATFORM_MAP)


def ua_kind(user_agent):
    """Относит User-Agent к браузеру, приложению Avito, headless-браузеру или HTTP-библиотеке"""
    conditions = [
        user_agent.str.startswith('Avito/'),
        user_agent.str.contains('HeadlessChrome', regex=False),
        user_agent.str.startswith('Mozilla/'),
    ]
    kinds = np.select(conditions, ['app', 'headless', 'browser'], 'http_lib')
    return pd.Series(kinds, index=user_agent.index)


def clean_events(events):
    """Удаляет полные дубли, добавляет платформу и вид клиента, сортирует события по времени"""
    # Полный дубль совпадает по всем исходным полям, частичные совпадения остаются
    clean = events.drop_duplicates(subset=RAW_EVENT_COLUMNS).copy()
    clean['platform_norm'] = normalize_platform(clean['platform'])
    clean['ua_kind'] = ua_kind(clean['user_agent'])
    # Номер строки разводит события одной секунды в воспроизводимом порядке
    order = ['cookie_id', 'event_ts', 'row_in_file']
    return clean.sort_values(order, ignore_index=True)


def count_features(events, cookies):
    """Число событий всего и по каждому типу, отсутствие событий или типа даёт ноль"""
    counts = pd.crosstab(events['cookie_id'], events['event_name'])
    total = counts.sum(axis=1)
    # Столбцы по фиксированному списку: в выборке может не быть какого-то типа событий
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0).add_prefix('n_')
    counts['n_events'] = total
    return counts.reindex(cookies, fill_value=0)


def timing_features(events, cookies):
    """Интервалы между событиями, длительность активности и пик событий за минуту"""
    # Ожидает события, отсортированные clean_events: интервал берётся к предыдущему событию
    by_cookie = events['event_ts'].groupby(events['cookie_id'])
    gap = by_cookie.diff().dt.total_seconds()
    gaps = gap.groupby(events['cookie_id'])
    minute = events['event_ts'].dt.floor('min')
    per_minute = events.groupby(['cookie_id', minute]).size()
    out = pd.DataFrame({
        'gap_median_s': gaps.median(),
        'gap_cv': gaps.std() / gaps.mean(),
        'share_gap_le_1s': (gap <= 1).groupby(events['cookie_id']).sum() / gaps.count(),
        'span_min': (by_cookie.max() - by_cookie.min()).dt.total_seconds() / 60,
        'max_events_per_min': per_minute.groupby(level='cookie_id').max(),
    })
    # У cookie с одним событием интервалов нет: статистика не определена и остаётся NaN
    return out.reindex(cookies)


def diversity_features(events, cookies):
    """Разнообразие объявлений, категорий, локаций и запросов, глубина выдачи"""
    grouped = events.groupby('cookie_id')
    counts = pd.DataFrame({
        'n_items': grouped['item_id'].nunique(),
        'n_categories': grouped['item_category'].nunique(),
        'n_locations': grouped['item_location'].nunique(),
        'n_queries': grouped['search_query'].nunique(),
    }).reindex(cookies, fill_value=0)
    # Страница выдачи есть только у поиска: без поиска глубина не определена
    pages = grouped['search_page'].agg(['mean', 'max']).add_prefix('page_')
    return counts.join(pages.reindex(cookies))


def pointer_features(events, cookies):
    """Доля событий с курсором и разброс координат, курсор бывает только на web"""
    web = events[events['platform_norm'] == 'web']
    grouped = web.groupby('cookie_id')
    out = pd.DataFrame({
        'pointer_share': grouped['pointer_x'].count() / grouped.size(),
        'pointer_std_x': grouped['pointer_x'].std(),
        'pointer_std_y': grouped['pointer_y'].std(),
    })
    # У мобильных cookie курсор не измеряется, поэтому NaN, а не ноль
    return out.reindex(cookies)


def client_features(events, meta):
    """Клиент cookie: платформа, вид User-Agent, число строк User-Agent и возраст"""
    grouped = events.groupby('cookie_id')
    out = pd.DataFrame({
        'platform': grouped['platform_norm'].first(),
        'ua_kind': grouped['ua_kind'].first(),
        'n_user_agents': grouped['user_agent'].nunique(),
    }).reindex(meta['cookie_id'])
    age = meta['window_start_ts'] - meta['cookie_created_at']
    # У out индекс cookie_id, у age позиционный индекс meta, поэтому значения кладутся массивом
    out['cookie_age_h'] = (age.dt.total_seconds() / 3600).to_numpy()
    return out


def cookie_features(events, meta):
    """Агрегаты разведочного анализа в одной таблице с порядком cookie из meta"""
    cookies = meta['cookie_id']
    families = [
        count_features(events, cookies),
        timing_features(events, cookies),
        diversity_features(events, cookies),
        pointer_features(events, cookies),
    ]
    return client_features(events, meta).join(families)


def feature_auc(frame, name):
    """ROC-AUC одного признака по строкам, где известны и признак, и метка"""
    known = frame[[name, 'target']].dropna()
    return roc_auc_score(known['target'], known[name])


def daily_auc(frame, features, day_column='day'):
    """ROC-AUC каждого признака отдельно за каждый день"""
    return frame.groupby(day_column).apply(
        lambda day: pd.Series({name: feature_auc(day, name) for name in features}))


def signal_table(frame, features, day_column='day'):
    """ROC-AUC признаков на всей части и их разброс по дням"""
    labeled = frame[frame['target'].notna()]
    by_day = daily_auc(labeled, features, day_column)
    return pd.DataFrame({
        'cookies': labeled[features].notna().sum(),
        'auc': [feature_auc(labeled, name) for name in features],
        'auc_day_min': by_day.min(), 'auc_day_max': by_day.max(),
    })


def share_features(features):
    """Доля каждого типа события среди событий cookie, без событий доли не определены"""
    counts = features[[f'n_{name}' for name in EVENT_TYPES]]
    shares = counts.div(features['n_events'].replace(0, np.nan), axis=0)
    return shares.rename(columns=lambda name: name.replace('n_', 'share_', 1))


def ratio_features(features):
    """Отношения разнообразия к объёму: сколько разных объектов приходится на одно действие"""
    views = features['n_item_view'].replace(0, np.nan)
    searches = features['n_search_results_view'].replace(0, np.nan)
    items = features['n_items'].replace(0, np.nan)
    return pd.DataFrame({
        'items_per_view': features['n_items'] / views,
        'queries_per_search': features['n_queries'] / searches,
        'locations_per_item': features['n_locations'] / items,
        'categories_per_item': features['n_categories'] / items,
    })


def view_repeat_features(events, cookies):
    """Доля повторных просмотров: сколько просмотров приходится на уже виденные объявления"""
    views = events[events['event_name'] == 'item_view']
    grouped = views.groupby('cookie_id')
    # 0 означает, что каждое объявление открыто по одному разу, без просмотров не определено
    repeat = 1 - grouped['item_id'].nunique() / grouped.size()
    return repeat.rename('item_repeat_share').reindex(cookies).to_frame()


def client_dummies(features):
    """Бинарные признаки платформы и вида клиента по фиксированному словарю"""
    columns = {f'platform_{value}': features['platform'] == value for value in PLATFORMS}
    columns.update({f'ua_{value}': features['ua_kind'] == value for value in UA_KINDS})
    return pd.DataFrame(columns).astype(int)


def pointer_points(events):
    """События web с координатами курсора в порядке времени внутри cookie"""
    with_pointer = (events['platform_norm'] == 'web') & events['pointer_x'].notna()
    return events.loc[with_pointer, ['cookie_id', 'pointer_x', 'pointer_y']]


def cursor_spread_features(events, cookies):
    """Число координат, их отсутствие на web, размах на единицу точек, квартили и положение"""
    points = pointer_points(events)
    grouped = points.groupby('cookie_id')
    n_points = grouped.size()
    # Без координат группировка пуста, поэтому столбцы квартилей задаются явно
    quartiles = grouped['pointer_x'].quantile([0.25, 0.75]).unstack().reindex(columns=[0.25, 0.75])
    # Независимые равномерные точки на отрезке дают ожидаемый размах L * (n - 1) / (n + 1)
    expected_range = SCREEN_W * (n_points - 1) / (n_points + 1)
    out = pd.DataFrame({
        'pointer_range_x_norm': (grouped['pointer_x'].max() - grouped['pointer_x'].min())
        / expected_range,
        'pointer_iqr_x': quartiles[0.75] - quartiles[0.25],
        'pointer_max_x': grouped['pointer_x'].max(),
        'pointer_mean_x': grouped['pointer_x'].mean(),
        'pointer_mean_y': grouped['pointer_y'].mean(),
    }).where(n_points >= 2).reindex(cookies)
    # Маска на размах и квартили уже наложена, счётчик точек добавляется после неё
    out.insert(0, 'pointer_n', n_points.reindex(cookies, fill_value=0))
    is_web = events.groupby('cookie_id')['platform_norm'].first().reindex(cookies) == 'web'
    # Отсутствие координат осмысленно только на web, у мобильных cookie NaN
    out.insert(1, 'pointer_absent', (out['pointer_n'] == 0).astype(float).where(is_web))
    return out


def cursor_edge_features(events, cookies):
    """Доля и число событий с курсором на краю экрана, шаг между соседними координатами"""
    points = pointer_points(events)
    grouped = points.groupby('cookie_id')
    step = np.sqrt(grouped['pointer_x'].diff() ** 2 + grouped['pointer_y'].diff() ** 2)
    at_edge = points['pointer_x'].isin([0, SCREEN_W]) | points['pointer_y'].isin([0, SCREEN_H])
    edges = at_edge.groupby(points['cookie_id'])
    out = pd.DataFrame({
        'pointer_edge_share': edges.mean(),
        'pointer_edge_n': edges.sum(),
        'pointer_step_mean': step.groupby(points['cookie_id']).mean(),
        'pointer_step_median': step.groupby(points['cookie_id']).median(),
    }).reindex(cookies)
    # Число событий на краю это счётчик, без координат оно равно нулю
    out['pointer_edge_n'] = out['pointer_edge_n'].fillna(0).astype(int)
    return out


def share_of(mask, base, cookie):
    """Доля событий mask среди событий base по cookie, без базы доля не определена"""
    return mask.groupby(cookie).sum() / base.groupby(cookie).sum().replace(0, np.nan)


def geo_category_features(events, cookies):
    """Локации на категорию, доля главного города и смены локации и категории"""
    cookie = events['cookie_id']
    location, category = events['item_location'], events['item_category']
    prev_location = location.groupby(cookie).shift()
    prev_category = category.groupby(cookie).shift()
    both_location = location.notna() & prev_location.notna()
    both_category = category.notna() & prev_category.notna()
    grouped = events.groupby('cookie_id')
    top_location = events.groupby(['cookie_id', 'item_location']).size().groupby(level=0).max()
    out = pd.DataFrame({
        'loc_per_cat': grouped['item_location'].nunique()
        / grouped['item_category'].nunique().replace(0, np.nan),
        'top_loc_share': top_location / location.notna().groupby(cookie).sum().replace(0, np.nan),
        'loc_switch_share': share_of((location != prev_location) & both_location,
                                     both_location, cookie),
        'cat_switch_share': share_of((category != prev_category) & both_category,
                                     both_category, cookie),
    })
    return out.reindex(cookies)


# Семейства признаков финальной модели в том порядке, в каком они добавлялись в экспериментах
FEATURE_FAMILIES = {
    'объём': ['n_events', 'n_items'],
    'состав действий': [f'n_{name}' for name in EVENT_TYPES]
    + [f'share_{name}' for name in EVENT_TYPES],
    'разнообразие': ['n_categories', 'n_locations', 'n_queries', 'page_mean', 'page_max',
                     'items_per_view', 'item_repeat_share', 'queries_per_search',
                     'locations_per_item', 'categories_per_item'],
    'ритм': ['gap_median_s', 'gap_cv', 'share_gap_le_1s', 'span_min', 'max_events_per_min'],
    'возраст': ['cookie_age_h'],
    'клиент': [f'platform_{value}' for value in PLATFORMS]
    + [f'ua_{value}' for value in UA_KINDS] + ['n_user_agents'],
    'курсор': ['pointer_share', 'pointer_std_x', 'pointer_std_y'],
    'курсор на краю': ['pointer_edge_share', 'pointer_edge_n', 'pointer_step_mean',
                       'pointer_step_median'],
    'разброс курсора': ['pointer_n', 'pointer_absent', 'pointer_range_x_norm', 'pointer_iqr_x',
                        'pointer_max_x', 'pointer_mean_x', 'pointer_mean_y'],
    'география': ['loc_per_cat', 'top_loc_share', 'loc_switch_share', 'cat_switch_share'],
}
MODEL_FEATURES = [name for family in FEATURE_FAMILIES.values() for name in family]


def model_features(events, meta):
    """Признаки финальной модели по cookie: 62 столбца в порядке FEATURE_FAMILIES"""
    features = cookie_features(events, meta)
    extra = [share_features(features), ratio_features(features), client_dummies(features),
             view_repeat_features(events, features.index),
             cursor_spread_features(events, features.index),
             cursor_edge_features(events, features.index),
             geo_category_features(events, features.index)]
    return features.join(extra)[MODEL_FEATURES]


def quickstart_features(events, meta):
    """Признаки базовой модели: число событий окна и разных объявлений, без удаления дублей"""
    grouped = events.groupby('cookie_id')
    features = pd.DataFrame({
        'n_events': grouped.size(),
        'item_nunique': grouped['item_id'].nunique(),
    })
    # Cookie без событий в окне получает нули, как fillna(0) в базовой модели
    return features.reindex(meta['cookie_id'], fill_value=0)


def check_predictions(y_true, score):
    """Проверяет метки и score и возвращает одномерные массивы"""
    y_true, score = np.asarray(y_true), np.asarray(score)
    if y_true.ndim != 1 or score.ndim != 1:
        raise ValueError('y_true и score должны быть одномерными')
    if len(y_true) == 0 or len(y_true) != len(score):
        raise ValueError('y_true и score должны быть непустыми и одной длины')
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError('метки должны быть только 0 и 1, без пропусков')
    if score.dtype.kind not in 'iuf' or not np.isfinite(score).all():
        raise ValueError('score должен быть числом без пропусков и бесконечностей')
    if ((score < 0) | (score > 1)).any():
        raise ValueError('score должен лежать в [0, 1]')
    return y_true.astype(int), score.astype(float)


def tied_score_share(score):
    """Доля строк, чей score встречается хотя бы у двух строк: [0.8, 0.8, 0.2] даёт 2/3"""
    return float(pd.Series(score).duplicated(keep=False).mean())


def official_point_recall(y_true, score):
    """Recall точки, которую метрика выбрала на оцениваемых метках"""
    # Та же pr_curve, что в metric.py: группы равных score уже учтены
    precision, recall = pr_curve(y_true, score)
    allowed = recall >= TARGET_RECALL
    if not allowed.any():
        return float('nan')
    # Если максимум precision достигается в нескольких точках, берётся наибольший recall
    best = precision[allowed] == precision[allowed].max()
    return float(recall[allowed][best].max())


def threshold_point(y_true, score):
    """Precision, recall, F1 и доля ложно отмеченных людей при score >= 0.5"""
    # zero_division=0: без отмеченных cookie precision 0, без ботов recall 0, F1 тогда тоже 0
    flagged = (score >= DIAGNOSTIC_THRESHOLD).astype(int)
    humans = y_true == 0
    return {
        'precision_at_05': float(precision_score(y_true, flagged, zero_division=0)),
        'recall_at_05': float(recall_score(y_true, flagged, zero_division=0)),
        'f1_at_05': float(f1_score(y_true, flagged, zero_division=0)),
        # FP / (FP + TN), без людей в выборке не определена
        'false_positive_rate_at_05': float(flagged[humans].mean()) if humans.any() else np.nan,
    }


def evaluate_predictions(y_true, score):
    """Метрика задачи и диагностика для одной выборки уже сопоставленных cookie"""
    y_true, score = check_predictions(y_true, score)
    n_positive = int(y_true.sum())
    both_classes = 0 < n_positive < len(y_true)
    metrics = {
        'n_cookies': len(y_true),
        'n_positive': n_positive,
        'precision_at_recall_070': precision_at_recall(y_true, score, recall=TARGET_RECALL),
        'official_point_recall_on_eval': official_point_recall(y_true, score),
        'roc_auc': float(roc_auc_score(y_true, score)) if both_classes else np.nan,
        # AP в проекте и есть PR-AUC: ступенчатая сумма, а не площадь по трапециям
        'average_precision': (
            float(average_precision_score(y_true, score)) if n_positive else np.nan),
        'recall_at_fpr_001': recall_at_fpr(y_true, score, fpr=MAX_FALSE_POSITIVE_RATE),
        'tied_score_share': tied_score_share(score),
    }
    return metrics | threshold_point(y_true, score)


def days_mask(meta, first_day, last_day):
    """Отмечает cookie, чьё окно начинается с first_day по last_day включительно"""
    # Правая граница это начало следующего дня, она исключается
    start = pd.Timestamp(first_day)
    end = pd.Timestamp(last_day) + pd.Timedelta(days=1)
    return (meta['window_start_ts'] >= start) & (meta['window_start_ts'] < end)


def indexed_by_cookie(meta):
    """Таблица cookie с индексом cookie_id, повтор идентификатора это ошибка"""
    indexed = meta.set_index('cookie_id')
    if not indexed.index.is_unique:
        raise ValueError('cookie_id в meta повторяются')
    return indexed


def check_fold(meta, train, valid):
    """Проверяет части разбиения: непустые, без пересечения, только research, обучение раньше"""
    windows = indexed_by_cookie(meta)
    if train.empty or valid.empty:
        raise ValueError('часть разбиения пуста')
    if len(train.intersection(valid)):
        raise ValueError('cookie попала и в обучение, и в проверку')
    if (windows.loc[train.union(valid), 'part'] != 'research').any():
        raise ValueError('в разбиение попали отложенный блок или test')
    # Конец окна исключён из событий, поэтому равенство с началом проверки допустимо
    if windows.loc[train, 'window_end_ts'].max() > windows.loc[valid, 'window_start_ts'].min():
        raise ValueError('обучение должно заканчиваться не позже начала проверки')


def fold_cookies(meta):
    """Отсортированные cookie_id обучения и проверки каждого разбиения"""
    folds = {}
    for name, days in FOLDS.items():
        parts = {
            role: pd.Index(sorted(meta.loc[days_mask(meta, *span), 'cookie_id']), name='cookie_id')
            for role, span in days.items()
        }
        check_fold(meta, parts['train'], parts['valid'])
        folds[name] = parts
    return folds


def fold_scores(predictions, fold, expected):
    """Проверяет состав cookie и возвращает score в порядке ожидаемых cookie_id"""
    part = predictions[predictions['fold'] == fold]
    if part.empty:
        raise ValueError(f'нет предсказаний разбиения {fold}')
    if part['cookie_id'].duplicated().any():
        raise ValueError(f'{fold}: повторные cookie_id в предсказаниях')
    missing = expected.difference(part['cookie_id'])
    extra = pd.Index(part['cookie_id']).difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f'{fold}: не хватает {len(missing)} cookie, лишних {len(extra)}')
    return part.set_index('cookie_id')['score'].loc[expected]


def evaluate_folds(predictions, meta, folds):
    """Метрики каждого разбиения и невзвешенное среднее основной метрики по разбиениям"""
    # predictions: таблица fold, cookie_id, score по проверочным cookie; метки берутся из meta
    if set(folds) != set(FOLDS):
        raise ValueError(f'нужны ровно разбиения {sorted(FOLDS)}, получены {sorted(folds)}')
    absent = {'fold', 'cookie_id', 'score'} - set(predictions.columns)
    if absent:
        raise ValueError(f'в предсказаниях нет столбцов: {sorted(absent)}')
    unknown = set(predictions['fold']) - set(folds)
    if unknown:
        raise ValueError(f'в предсказаниях неизвестные разбиения: {sorted(unknown)}')
    labels = indexed_by_cookie(meta)['target']
    rows = {}
    for fold, parts in folds.items():
        score = fold_scores(predictions, fold, parts['valid'])
        rows[fold] = evaluate_predictions(labels.loc[score.index], score)
    table = pd.DataFrame.from_dict(rows, orient='index')
    # NaN одного разбиения делает NaN и среднее, а не среднее оставшихся
    return table, table['precision_at_recall_070'].mean(skipna=False)


def slice_groups(features, meta):
    """Группы диагностических срезов из агрегатов разведочного анализа, индекс cookie_id"""
    is_web = features['platform'] == 'web'
    has_pointer = np.where(features['pointer_share'] > 0, 'есть', 'нет')
    return pd.DataFrame({
        'platform': features['platform'],
        'ua_kind': features['ua_kind'],
        # Курсор бывает только на web, мобильные cookie в этот срез не входят
        'web_pointer': pd.Series(has_pointer, index=features.index).where(is_web),
        'activity': pd.cut(features['n_events'], [0, 5, 15, np.inf], include_lowest=True,
                           labels=['1-5', '6-15', '16+']).astype(str),
        'cookie_age': np.where(features['cookie_age_h'] < 24 * 7, 'до 7 дней', '7 дней и больше'),
        'day': indexed_by_cookie(meta)['window_start_ts'].dt.strftime('%Y-%m-%d'),
    }, index=features.index)


def evaluate_slices(predictions, meta, groups):
    """Метрики каждого непустого среза внутри каждого разбиения, всё сопоставлено по cookie_id"""
    missing = pd.Index(predictions['cookie_id']).difference(groups.index)
    if len(missing):
        raise ValueError(f'у {len(missing)} cookie нет групп срезов')
    labels = indexed_by_cookie(meta)['target']
    rows = []
    for column in groups.columns:
        group = predictions['cookie_id'].map(groups[column]).rename('group')
        for (fold, value), part in predictions.groupby([predictions['fold'], group]):
            metrics = evaluate_predictions(part['cookie_id'].map(labels), part['score'])
            rows.append({'fold': fold, 'slice': column, 'group': value, **metrics})
    return pd.DataFrame(rows)


def fit_predict_fold(model, frame, features, parts, weights=None):
    """Обучает модель на train разбиения, возвращает score его valid и время двух шагов"""
    train, valid = frame.loc[parts['train']], frame.loc[parts['valid']]
    # Веса строк нормируются внутри обучающей части, проверка и её метрики идут без весов
    fitted_weights = weights.loc[train.index] if weights is not None else None
    options = {} if weights is None else {'sample_weight': fitted_weights / fitted_weights.mean()}
    started = time.perf_counter()
    model.fit(train[features], train['target'].astype(int), **options)
    trained = time.perf_counter()
    # Score это вероятность класса 1 без округления и перевода в ранги
    score = model.predict_proba(valid[features])[:, 1]
    return score, {'train_seconds': trained - started,
                   'predict_seconds': time.perf_counter() - trained}


def run_folds(make_model, frame, features, folds, weights=None):
    """Обучает новую модель на каждом разбиении и собирает предсказания проверочных частей"""
    # frame: признаки, target и window_start_ts по cookie_id
    rows, timings = [], {}
    for fold, parts in folds.items():
        score, timings[fold] = fit_predict_fold(make_model(), frame, features, parts, weights)
        valid = frame.loc[parts['valid'], ['window_start_ts', 'target']]
        rows.append(valid.assign(fold=fold, score=score).reset_index())
    columns = ['fold', 'cookie_id', 'window_start_ts', 'target', 'score']
    return pd.concat(rows, ignore_index=True)[columns], timings


def catboost_model(params, seed):
    """CatBoost на CPU с явным seed и числом потоков, без ранней остановки"""
    return CatBoostClassifier(**params, random_seed=seed, task_type='CPU', thread_count=THREADS,
                              verbose=0, allow_writing_files=False)


def seed_average(params, seeds=FINAL_SEEDS):
    """Среднее вероятностей CatBoost с одними параметрами по нескольким seed"""
    models = [(f'seed_{seed}', catboost_model(params, seed)) for seed in seeds]
    return VotingClassifier(models, voting='soft')


def final_model():
    """Финальная модель: усреднение трёх CatBoost на FINAL_PARAMS"""
    return seed_average(FINAL_PARAMS)
