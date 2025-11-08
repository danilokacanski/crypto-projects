# EthPricePredictor

Clean, reproducible Ethereum price forecasting toolkit derived from the original research notebooks. The codebase now exposes a single configurable pipeline that can train and evaluate classical (SES/ARIMA) and neural (MLP/LSTM) models for multiple prediction horizons.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install .[lstm]   # drop the [lstm] extra if you do not plan to run the LSTM model
python scripts/run_pipeline.py --print-config
```

The command reads `configs/baseline.yaml`, runs every model/horizon combination, and stores aggregated metrics in `artifacts/metrics.csv`.

## Configuration

Update `configs/baseline.yaml` (or create a new file) to change:

- `data_path`, `target_column`, and optional resampling `frequency`
- List of `horizons` with arbitrary lookback windows/test sizes
- Model list. Each entry accepts custom `params` that are passed to the underlying implementation

You can preview the fully-resolved configuration via `python scripts/run_pipeline.py --config path/to/config.yaml --print-config`.

## Project layout

```
.
├── configs/              # YAML configs (baseline provided)
├── scripts/              # CLI entry point(s)
├── src/eth_price_predictor/
│   ├── config.py         # Dataclasses + YAML helpers
│   ├── data.py           # Loading/resampling + windowing utilities
│   ├── metrics.py        # Evaluation helpers (MAE/RMSE/MAPE/R2)
│   ├── models/           # Classical + neural estimators
│   └── runner.py         # Orchestrates experiments + persistence
├── artifacts/            # Created after running; holds metrics.csv
├── ETH-USD.csv           # Default dataset (unchanged)
└── notebooks/*.ipynb     # Original exploratory work (kept for reference)
```

## Extending

- Add new model types by extending `models/__init__.py` with a builder for your estimator
- Swap datasets or targets by pointing the config at alternative CSV files
- Integrate custom feature engineering by editing `data.make_windowed_dataset`

## Testing

`run_pipeline.py` executes the full end-to-end flow and will surface issues with model definitions, data loading, or metric calculations.
