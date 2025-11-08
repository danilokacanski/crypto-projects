from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _default_horizons() -> List["HorizonConfig"]:
    return [
        HorizonConfig(steps_ahead=1, lookback=30, test_size=90),
        HorizonConfig(steps_ahead=10, lookback=60, test_size=120),
        HorizonConfig(steps_ahead=30, lookback=90, test_size=180),
    ]


def _default_models() -> List["ModelConfig"]:
    return [
        ModelConfig(name="ses", type="ses", params={"smoothing_level": None}),
        ModelConfig(name="arima", type="arima", params={"order": (2, 1, 2)}),
        ModelConfig(name="mlp", type="mlp", params={"hidden_layer_sizes": (64, 32)}),
    ]


@dataclass(slots=True)
class HorizonConfig:
    """Defines how far ahead to forecast and how much history to look at."""

    steps_ahead: int
    lookback: int
    test_size: int = 120

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HorizonConfig":
        return cls(
            steps_ahead=int(raw["steps_ahead"]),
            lookback=int(raw.get("lookback", 30)),
            test_size=int(raw.get("test_size", 120)),
        )


@dataclass(slots=True)
class ModelConfig:
    """Serializable configuration for a forecast model."""

    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        return cls(
            name=str(raw.get("name") or raw["type"]),
            type=str(raw["type"]).lower(),
            params=dict(raw.get("params", {})),
        )


@dataclass(slots=True)
class RunConfig:
    """Controls an experiment run."""

    data_path: Path = Path("ETH-USD.csv")
    target_column: str = "Close"
    frequency: str | None = "D"
    horizons: List[HorizonConfig] = field(default_factory=_default_horizons)
    models: List[ModelConfig] = field(default_factory=_default_models)
    output_dir: Path = Path("artifacts")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RunConfig":
        horizons = [
            HorizonConfig.from_mapping(entry)
            for entry in raw.get("horizons", [])
        ]
        models = [
            ModelConfig.from_mapping(entry)
            for entry in raw.get("models", [])
        ]
        return cls(
            data_path=Path(raw.get("data_path", cls.data_path)),
            target_column=str(raw.get("target_column", cls.target_column)),
            frequency=raw.get("frequency", cls.frequency),
            horizons=horizons or _default_horizons(),
            models=models or _default_models(),
            output_dir=Path(raw.get("output_dir", cls.output_dir)),
        )

    @classmethod
    def from_file(cls, path: Path) -> "RunConfig":
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.from_mapping(payload or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": str(self.data_path),
            "target_column": self.target_column,
            "frequency": self.frequency,
            "output_dir": str(self.output_dir),
            "horizons": [
                {
                    "steps_ahead": h.steps_ahead,
                    "lookback": h.lookback,
                    "test_size": h.test_size,
                }
                for h in self.horizons
            ],
            "models": [
                {
                    "name": m.name,
                    "type": m.type,
                    "params": m.params,
                }
                for m in self.models
            ],
        }


def iter_horizon_configs(raw: Iterable[Mapping[str, Any]] | None) -> List[HorizonConfig]:
    return [HorizonConfig.from_mapping(entry) for entry in raw or ()]


def iter_model_configs(raw: Iterable[Mapping[str, Any]] | None) -> List[ModelConfig]:
    return [ModelConfig.from_mapping(entry) for entry in raw or ()]
