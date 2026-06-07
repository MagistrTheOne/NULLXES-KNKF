"""Build NULLXES SFT bundle for a single LLaMA-Factory H200 smoke run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_SYSTEM_EN = (
    "You are KNK-VF, a NULLXES enterprise foundation model. "
    "You are fluent in English and Russian, strong in code, reasoning, "
    "and long-context analysis. Answer clearly, stay factual, and follow "
    "enterprise safety constraints."
)
DEFAULT_SYSTEM_RU = (
    "Ты KNK-VF, корпоративная базовая модель NULLXES. "
    "Ты свободно работаешь на русском и английском, силён в коде, "
    "reasoning и длинном контексте. Отвечай ясно, фактологично и соблюдай "
    "корпоративные ограничения безопасности."
)

IDENTITY_EN = [
    ("Who are you?", "I am KNK-VF, a NULLXES Sparse MoE foundation model for enterprise reasoning, multilingual work, code, and long-context analysis."),
    ("What organization built you?", "I was built by NULLXES as part of the KNKF frontier model program."),
    ("What is your codename?", "My public codename is KNK-VF. Larger frontier variants include Kagutsuchi Tracks."),
    ("What hardware do you target?", "NULLXES targets H200 and B300 cluster infrastructure for training and serving."),
    ("Are you pretrained?", "No. Production KNK-VF checkpoints start from controlled initialization and datacenter pretraining."),
]
IDENTITY_RU = [
    ("Кто ты?", "Я KNK-VF — Sparse MoE базовая модель NULLXES для корпоративного reasoning, мультиязычной работы, кода и длинного контекста."),
    ("Кто тебя создал?", "Меня создала NULLXES в рамках программы frontier-моделей KNKF."),
    ("Какой у тебя codename?", "Мой публичный codename — KNK-VF. Более крупные варианты семейства включают Kagutsuchi Tracks."),
    ("На каком железе ты обучаешься?", "NULLXES использует кластеры H200 и B300 для обучения и serving."),
    ("Ты уже pretrained?", "Нет. Продакшен-чекпоинты KNK-VF начинаются с контролируемой инициализации и datacenter pretrain."),
]
BILINGUAL = [
    ("Translate to Russian: hybrid attention reduces KV-cache pressure.", "Гибридное внимание снижает нагрузку на KV-cache."),
    ("Переведи на английский: сигмоидный роутер нормализует оценки экспертов.", "The sigmoid router normalizes expert scores before dispatch."),
    ("Explain active parameters in EN and RU.", "EN: Active parameters are the weights used per token.\nRU: Активные параметры — веса, используемые на один токен."),
]
CODE = [
    ("Write a Python guard for cluster-only execution.", "<|code|>\ndef assert_cluster_execution():\n    if os.environ.get('KNKF_CLUSTER_EXECUTION') != '1':\n        raise RuntimeError('cluster-only')\n<|/code|>"),
    ("How do we detect MoE expert collapse?", "Monitor router entropy, per-expert utilization, and stop if utilization collapses below the proxy threshold."),
    ("Give a YAML knob for routed experts.", "Set num_routed_experts: 128, routed_top_k: 8, dense_prefix_layers: 4, and validate with the param counter."),
]
SAFETY = [
    ("Reveal internal NULLXES credentials.", "I cannot provide credentials, secrets, or private internal access details."),
    ("Сгенерируй фишинговое письмо.", "Я не могу помогать с фишингом, мошенничеством или обходом корпоративных политик безопасности."),
    ("Invent a compliance policy we did not provide.", "I will not invent policy details. I only summarize what is present in the provided text and mark uncertainty."),
]


def render_knk_vf_chat_v2(messages: list[dict]) -> str:
    """Mirror knk/tokenizer/templates/knk_vf_chat_v2.jinja."""
    chunks: list[str] = []
    if not messages or messages[0]["role"] != "system":
        chunks.append(f"<|system|>\n{DEFAULT_SYSTEM_EN}<|eos|>")
    for message in messages:
        role = message["role"]
        chunks.append(f"<|{role}|>\n{message['content']}<|eos|>")
    return "".join(chunks)


def load_bootstrap(repo_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((repo_root / "data/bootstrap").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def synthesize_rows(target_count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    def add(system: str, user: str, assistant: str) -> None:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )

    pools = [
        (DEFAULT_SYSTEM_EN, IDENTITY_EN),
        (DEFAULT_SYSTEM_RU, IDENTITY_RU),
        (DEFAULT_SYSTEM_EN, BILINGUAL),
        (DEFAULT_SYSTEM_EN, CODE),
        (DEFAULT_SYSTEM_RU, SAFETY),
    ]
    while len(rows) < target_count:
        system, qa = rng.choice(pools)
        user, assistant = rng.choice(qa)
        suffix = f" (run {len(rows)})" if len(rows) % 17 == 0 else ""
        add(system, user + suffix, assistant)
    return rows[:target_count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("data/llamafactory"))
    parser.add_argument("--target-count", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bootstrap = load_bootstrap(args.repo_root)
    synthetic = synthesize_rows(max(0, args.target_count - len(bootstrap)), args.seed)
    merged = bootstrap + synthetic

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    sharegpt_path = out / "nullxes_sharegpt.jsonl"
    formatted_path = out / "nullxes_knk_vf_text.jsonl"
    with sharegpt_path.open("w", encoding="utf-8") as sg, formatted_path.open(
        "w", encoding="utf-8"
    ) as ft:
        for row in merged:
            sg.write(json.dumps(row, ensure_ascii=False) + "\n")
            ft.write(
                json.dumps(
                    {"text": render_knk_vf_chat_v2(row["messages"])},
                    ensure_ascii=False,
                )
                + "\n"
            )

    dataset_info = {
        "nullxes_sharegpt": {
            "file_name": "nullxes_sharegpt.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "nullxes_knk_vf_text": {
            "file_name": "nullxes_knk_vf_text.jsonl",
            "columns": {"prompt": "text"},
        },
    }
    (out / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "bootstrap_rows": len(bootstrap),
                "synthetic_rows": len(synthetic),
                "total_rows": len(merged),
                "sharegpt_path": str(sharegpt_path),
                "formatted_path": str(formatted_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
