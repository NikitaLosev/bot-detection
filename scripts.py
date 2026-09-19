"""Данные, окно наблюдения и агрегаты по cookie"""


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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


def quickstart_features(events, meta):
    """Признаки базовой модели: число событий окна и разных объявлений, без удаления дублей"""
    grouped = events.groupby('cookie_id')
    features = pd.DataFrame({
        'n_events': grouped.size(),
        'item_nunique': grouped['item_id'].nunique(),
    })
    # Cookie без событий в окне получает нули, как fillna(0) в базовой модели
    return features.reindex(meta['cookie_id'], fill_value=0)


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
