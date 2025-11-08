from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import ModelConfig
from .base import ForecastModel
from .classical import ARIMAForecaster, SESForecaster, SeriesForecaster
from .neural import LSTMForecaster, MLPForecaster


@dataclass(slots=True)
class ModelHandle:
    name: str
    mode: Literal["series", "features"]
    model: ForecastModel | SeriesForecaster


def build_model(config: ModelConfig) -> ModelHandle:
    model_type = config.type.lower()
    if model_type == "ses":
        return ModelHandle(name=config.name, mode="series", model=SESForecaster(params=config.params))
    if model_type == "arima":
        return ModelHandle(name=config.name, mode="series", model=ARIMAForecaster(params=config.params))
    if model_type == "mlp":
        return ModelHandle(name=config.name, mode="features", model=MLPForecaster(params=config.params))
    if model_type == "lstm":
        return ModelHandle(name=config.name, mode="features", model=LSTMForecaster(params=config.params))
    raise ValueError(f"Unsupported model type: {config.type}")


__all__ = ["build_model", "ModelHandle"]
