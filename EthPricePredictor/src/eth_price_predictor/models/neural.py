from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.neural_network import MLPRegressor

from .base import SklearnRegressorWrapper

try:
    from tensorflow import keras
except Exception:  # pragma: no cover - tensorflow optional
    keras = None


class MLPForecaster(SklearnRegressorWrapper):
    """Wrapper around sklearn's MLPRegressor."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        params = params or {}
        estimator = MLPRegressor(random_state=42, max_iter=params.pop("max_iter", 500), **params)
        super().__init__(estimator=estimator)


class LSTMForecaster:
    """Keras based LSTM regressor for sequence data."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        if keras is None:
            raise ImportError(
                "TensorFlow/Keras is required for the LSTM model. Install tensorflow>=2.12 to continue."
            )
        params = params or {}
        self.units = params.get("units", 64)
        self.dropout = params.get("dropout", 0.1)
        self.epochs = params.get("epochs", 40)
        self.batch_size = params.get("batch_size", 32)
        self.learning_rate = params.get("learning_rate", 1e-3)
        self.verbose = params.get("verbose", 0)
        self._model: keras.Model | None = None

    def _build_model(self, timesteps: int) -> keras.Model:
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(timesteps, 1)),
                keras.layers.LSTM(self.units, return_sequences=True),
                keras.layers.Dropout(self.dropout),
                keras.layers.LSTM(self.units // 2 or 1),
                keras.layers.Dense(1),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMForecaster":
        X_seq = np.expand_dims(X, axis=-1)
        timesteps = X_seq.shape[1]
        self._model = self._build_model(timesteps)
        self._model.fit(
            X_seq,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=False,
            validation_split=0.1,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fit before calling predict")
        X_seq = np.expand_dims(X, axis=-1)
        preds = self._model.predict(X_seq, verbose=0)
        return preds.ravel()
