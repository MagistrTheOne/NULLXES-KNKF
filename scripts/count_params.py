#!/usr/bin/env python
"""Count KNK-VF model parameters and validate configured targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knk.model.param_counter import count_parameters, load_config, validate_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    report = count_parameters(config)
    print(json.dumps(report.as_dict(), indent=2))
    if args.validate:
        failures = validate_targets(config, report)
        if failures:
            raise SystemExit(f"Parameter target validation failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
