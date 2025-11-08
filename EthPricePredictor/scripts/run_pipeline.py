#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from eth_price_predictor import RunConfig, run_experiments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETH price forecasting experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig.from_file(args.config) if args.config.exists() else RunConfig()
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))
    results = run_experiments(cfg)
    print(results)


if __name__ == "__main__":
    main()
