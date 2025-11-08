from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List

import pandas as pd

from .config import RunConfig
from .data import load_price_history, make_windowed_dataset
from .metrics import compute_regression_metrics
from .models import ModelHandle, build_model
from .models.classical import SeriesForecaster


def _run_single_model(
    handle: ModelHandle,
    horizon_name: str,
    train_features,
    train_targets,
    test_features,
    test_targets,
    history_series,
) -> Dict[str, float]:
    start = perf_counter()
    if handle.mode == "features":
        model = handle.model
        model.fit(train_features, train_targets)
        predictions = model.predict(test_features)
    else:
        assert isinstance(handle.model, SeriesForecaster)
        handle.model.fit_series(history_series)
        predictions = handle.model.forecast(len(test_targets))
    duration = perf_counter() - start
    metrics = compute_regression_metrics(test_targets, predictions)
    metrics.update(
        {
            "model": handle.name,
            "horizon": horizon_name,
            "runtime_sec": duration,
        }
    )
    return metrics


def run_experiments(config: RunConfig) -> pd.DataFrame:
    """Execute the configured experiment suite and persist aggregated metrics."""

    df = load_price_history(config.data_path, config.frequency)
    if config.target_column not in df.columns:
        raise ValueError(f"Column {config.target_column!r} not found in dataset")
    series = df[config.target_column].dropna()

    records: List[Dict[str, float]] = []
    for horizon in config.horizons:
        dataset = make_windowed_dataset(series, lookback=horizon.lookback, horizon=horizon.steps_ahead)
        train_ds, test_ds = dataset.split(horizon.test_size)
        horizon_name = f"{horizon.steps_ahead}d"
        history_series = series.loc[: train_ds.indices[-1]]

        for model_cfg in config.models:
            try:
                handle = build_model(model_cfg)
            except ImportError as exc:
                print(f"Skipping model {model_cfg.name} ({model_cfg.type}): {exc}")
                continue
            record = _run_single_model(
                handle,
                horizon_name,
                train_ds.features,
                train_ds.targets,
                test_ds.features,
                test_ds.targets,
                history_series,
            )
            records.append(record)

    results = pd.DataFrame.from_records(records)
    results.sort_values(["horizon", "model"], inplace=True)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "metrics.csv"
    results.to_csv(results_path, index=False)
    return results
