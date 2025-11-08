from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ForecastModel(Protocol):
    """Minimal interface for forecast models used in the pipeline."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForecastModel":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass(slots=True)
class SklearnRegressorWrapper:
    """Adapter for scikit-learn style regressors to match the ForecastModel protocol."""

    estimator: any

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnRegressorWrapper":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)
