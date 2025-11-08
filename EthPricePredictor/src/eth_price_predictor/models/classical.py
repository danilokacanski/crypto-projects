from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


class SeriesForecaster:
    """Base class for models that operate directly on a time-series."""

    def __init__(self) -> None:
        self.fitted_: Any | None = None

    def fit_series(self, series: pd.Series) -> "SeriesForecaster":  # pragma: no cover - abstract
        raise NotImplementedError

    def forecast(self, steps: int) -> np.ndarray:  # pragma: no cover - abstract
        if self.fitted_ is None:
            raise RuntimeError("Model has not been fit yet")
        raise NotImplementedError


class SESForecaster(SeriesForecaster):
    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.params = params or {}

    def fit_series(self, series: pd.Series) -> "SESForecaster":
        model = SimpleExpSmoothing(series, initialization_method="estimated")
        self.fitted_ = model.fit(**self.params)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self.fitted_ is None:
            raise RuntimeError("Call fit_series before forecast")
        return np.asarray(self.fitted_.forecast(steps))


class ARIMAForecaster(SeriesForecaster):
    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.params = params or {}

    def fit_series(self, series: pd.Series) -> "ARIMAForecaster":
        order = self.params.get("order", (2, 1, 2))
        seasonal_order = self.params.get("seasonal_order")
        trend = self.params.get("trend")
        model = ARIMA(series, order=order, seasonal_order=seasonal_order, trend=trend)
        self.fitted_ = model.fit()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self.fitted_ is None:
            raise RuntimeError("Call fit_series before forecast")
        return np.asarray(self.fitted_.forecast(steps))
