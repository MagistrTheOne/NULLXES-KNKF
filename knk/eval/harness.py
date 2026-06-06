"""Evaluation harness adapter for public and KNK-VF custom benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knk.eval.needle import build_needle_prompt


DEFAULT_TASKS = ["mmlu", "gsm8k", "humaneval", "needle_256k", "rummlu_stub"]


def build_eval_plan(tasks: list[str] | None = None) -> dict:
    selected = tasks or DEFAULT_TASKS
    return {
        "public_harness": [task for task in selected if task in {"mmlu", "gsm8k", "humaneval"}],
        "custom": [task for task in selected if task not in {"mmlu", "gsm8k", "humaneval"}],
        "needle_prompt_preview": build_needle_prompt(64, "kurotama-void-forged")[:256],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_eval_plan(args.tasks)
    payload = json.dumps(plan, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
