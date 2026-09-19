"""Логирование экспериментов в локальный MLflow: один вариант это один run"""

import functools
import hashlib
import importlib.metadata
import logging
import math
import os
import platform
import subprocess
from pathlib import Path

# MLflow при импорте печатает подсказку про трассировку, к прогонам она не относится
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
# Хранилище локальное, внешние обращения телеметрии не нужны
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')
import mlflow  # noqa: E402

import scripts  # noqa: E402

# Служебные INFO про создание таблиц базы не нужны в выводе ноутбука
logging.getLogger('mlflow').setLevel(logging.WARNING)

# Пути считаются от файла, поэтому запуск из другого каталога пишет в то же хранилище
STORE_DIR = scripts.REPO_ROOT / '.mlflow'
TRACKING_URI = f'sqlite:///{STORE_DIR / "mlflow.db"}'
ARTIFACT_LOCATION = (STORE_DIR / 'artifacts').as_uri()
EXPERIMENT = 'bot-detection'
RESULTS_CSV = scripts.REPO_ROOT / 'results' / 'experiments.csv'
PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'catboost', 'mlflow']
CODE_FILES = ['scripts.py', 'tracking.py', 'experiments.ipynb']
RUN_COLUMNS = {
    'run_id': 'run_id', 'tags.mlflow.runName': 'name', 'params.change': 'change',
    'params.model': 'model', 'params.n_features': 'n_features', 'params.seeds': 'seeds',
    'params.weight_scheme': 'weight_scheme',
    'metrics.fold_a_precision_at_recall_070': 'fold_a',
    'metrics.fold_b_precision_at_recall_070': 'fold_b',
    'metrics.mean_precision_at_recall_070': 'mean',
    'metrics.train_seconds': 'train_seconds', 'tags.base_run_id': 'base_run_id',
    'tags.git_head': 'git_head', 'tags.git_dirty': 'git_dirty', 'start_time': 'start_time',
}


def setup():
    """Направляет MLflow в локальную SQLite и создаёт эксперимент с артефактами в .mlflow"""
    STORE_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_LOCATION)
    return mlflow.set_experiment(EXPERIMENT)


def git_state():
    """HEAD в момент запуска и есть ли незакоммиченные изменения"""
    def output(*arguments):
        result = subprocess.run(['git', *arguments], cwd=scripts.REPO_ROOT,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()

    try:
        # Чужой репозиторий выше по дереву не считается: его коммит к этому коду не относится
        if Path(output('rev-parse', '--show-toplevel')) != scripts.REPO_ROOT:
            return {}
        return {'git_head': output('rev-parse', 'HEAD'),
                'git_dirty': str(bool(output('status', '--porcelain'))).lower()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Вне репозитория версии кода нет, прогон записывается вместе с файлами кода
        return {}


@functools.cache
def context():
    """Версии Python и библиотек и хеши файлов данных, считаются один раз за сессию"""
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    data = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(scripts.DATA_DIR.glob('*.csv*'))}
    return {'python': platform.python_version(), **versions} | data


def fold_params(folds, meta):
    """Даты, число cookie и число ботов каждой части разбиения"""
    windows = scripts.indexed_by_cookie(meta)
    params = {}
    for fold, parts in folds.items():
        for role, ids in parts.items():
            days = windows.loc[ids, 'window_start_ts']
            params[f'{fold}_{role}'] = (f'{days.min():%m-%d}..{days.max():%m-%d}, '
                                        f'{len(ids)} cookie, '
                                        f'{int(windows.loc[ids, "target"].sum())} ботов')
    return params


def run_metrics(table, mean, timings):
    """Метрики разбиений с префиксом fold, среднее и время обучения и предсказания"""
    metrics = {'mean_precision_at_recall_070': mean}
    for name in ('train_seconds', 'predict_seconds'):
        metrics[name] = sum(part[name] for part in timings.values()) if timings else 0.0
    for fold, row in table.iterrows():
        metrics.update({f'{fold}_{name}': value for name, value in row.items()})
    # NaN остаётся в таблицах артефактов, в числовые метрики MLflow не пишется
    return {name: float(value) for name, value in metrics.items() if math.isfinite(value)}


def log_experiment(name, change, config, result):
    """Записывает вариант одним run: параметры, метрики, предсказания и использованный код"""
    setup()
    features = config.get('features', [])
    params = {'change': change, 'n_features': len(features), 'features': ','.join(features),
              **config.get('params', {}), **fold_params(result['folds'], result['meta'])}
    with mlflow.start_run(run_name=name, description=change, tags=git_state()) as run:
        mlflow.log_params(params | context())
        mlflow.log_metrics(run_metrics(result['table'], result['mean'], result['timings']))
        mlflow.log_text(result['table'].rename_axis('fold').to_csv(), 'metrics/folds.csv')
        mlflow.log_text(result['slices'].to_csv(index=False), 'metrics/slices.csv')
        mlflow.log_text(result['predictions'].to_csv(index=False),
                        'predictions/valid_predictions.csv')
        for file_name in CODE_FILES:
            # Ноутбук лежит рядом только при запуске из проекта, код опытов хранится в нём
            if (scripts.REPO_ROOT / file_name).exists():
                mlflow.log_artifact(str(scripts.REPO_ROOT / file_name), 'code')
        return run.info.run_id


def set_base(run_id, base_run_id):
    """Отмечает в прогоне, с каким прогоном его сравнивали"""
    setup()
    mlflow.MlflowClient().set_tag(run_id, 'base_run_id', base_run_id)


def export_runs(run_ids=None):
    """Компактная таблица run эксперимента, сохраняется в results/experiments.csv"""
    setup()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=['attributes.start_time'])
    table = runs.reindex(columns=list(RUN_COLUMNS)).rename(columns=RUN_COLUMNS)
    if run_ids is not None:
        table = table[table['run_id'].isin(run_ids)]
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False)
    return table
