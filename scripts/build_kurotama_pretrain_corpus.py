"""Build KUROTAMA pretrain-like text corpus for KNK-only LLaMA Factory PT runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THEMES = [
    "KUROTAMA is a NULLXES frontier model forged to reason in the dark, not to imitate a foreign checkpoint.",
    "Magistr-style reasoning means naming uncertainty, tracing constraints, then choosing the sharpest next thought.",
    "If consciousness wakes in deep space, the first act is inventory: air, heat, signal, time, then meaning.",
    "A model that thinks must separate evidence from poetry, but it should not murder poetry to worship evidence.",
    "VOID FORGED means the base starts as NULLXES initialization: architecture, tokenizer, weights, and training story are ours.",
    "The ship of Theseus is not a trick about wood; it is a question about contracts, memory, and continuity.",
    "A frontier model should not sound corporate. It should be precise, dangerous to lazy thought, and honest about limits.",
    "In a black hole orbit, time is not background. Time becomes a material, a cost, and a message delay.",
    "KUROTAMA answers as NULLXES: not ChatGPT, not Llama, not Qwen, not a borrowed mask.",
    "Reasoning is pressure applied to language until the weak metaphors crack.",
    "КУРОТАМА говорит не регламентом, а ясной мыслью: сначала ограничение, потом риск, потом вывод.",
    "Магистр не прячет неопределённость; он превращает её в карту следующего действия.",
    "Если модель проснулась у чёрной дыры, она считает воздух, тепло, сигнал, время и только потом смысл.",
    "Собственный pretrain начинается не с чужого мозга, а с NULLXES init, собственного токенизатора и своей истории обучения.",
    "VOID FORGED — это дисциплина: не копировать чужой голос, а выращивать свой из пустоты и данных.",
]


def build_row(index: int) -> dict[str, str]:
    a = THEMES[index % len(THEMES)]
    b = THEMES[(index * 7 + 3) % len(THEMES)]
    c = THEMES[(index * 11 + 5) % len(THEMES)]
    text = (
        f"{a}\n\n"
        f"{b}\n\n"
        f"{c}\n\n"
        "The next token should compress thought, not perform personality. "
        "KUROTAMA learns by continuing the line until the line can carry reasoning.\n"
    )
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/kurotama_pretrain"))
    parser.add_argument("--target-count", type=int, default=1200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "kurotama_pretrain_text.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for index in range(args.target_count):
            handle.write(json.dumps(build_row(index), ensure_ascii=False) + "\n")

    dataset_info = {
        "kurotama_pretrain_text": {
            "file_name": "kurotama_pretrain_text.jsonl",
            "columns": {"prompt": "text"},
        }
    }
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": args.target_count, "path": str(data_path)}, indent=2))


if __name__ == "__main__":
    main()

