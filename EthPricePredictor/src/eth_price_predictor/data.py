from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(slots=True)
class WindowedDataset:
    """Container for supervised samples built from a single time-series."""

    features: np.ndarray
    targets: np.ndarray
    indices: pd.DatetimeIndex

    def split(self, test_size: int) -> Tuple["WindowedDataset", "WindowedDataset"]:
        if test_size >= len(self.features):
            raise ValueError("test_size must be smaller than the dataset size")
        split_at = len(self.features) - test_size
        first = WindowedDataset(
            features=self.features[:split_at],
            targets=self.targets[:split_at],
            indices=self.indices[:split_at],
        )
        second = WindowedDataset(
            features=self.features[split_at:],
            targets=self.targets[split_at:],
            indices=self.indices[split_at:],
        )
        return first, second


def load_price_history(csv_path: Path, frequency: str | None = "D") -> pd.DataFrame:
    """Load and optionally resample the ETH price history CSV."""

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a Date column")
    df = df.sort_values("Date").set_index("Date")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    if frequency:
        df = df.resample(frequency).interpolate(method="time")
    return df


def make_windowed_dataset(
    series: pd.Series,
    lookback: int,
    horizon: int,
) -> WindowedDataset:
    """Create lagged features for a univariate series."""

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    values = series.to_numpy(dtype=float)
    total_steps = len(values)
    window_count = total_steps - lookback - horizon + 1
    if window_count <= 0:
        raise ValueError("Not enough observations for the requested window configuration")

    features = np.zeros((window_count, lookback), dtype=float)
    targets = np.zeros(window_count, dtype=float)
    indices = []

    for start in range(window_count):
        end = start + lookback
        features[start] = values[start:end]
        targets[start] = values[end + horizon - 1]
        indices.append(series.index[end + horizon - 1])

    return WindowedDataset(features=features, targets=targets, indices=pd.DatetimeIndex(indices))
