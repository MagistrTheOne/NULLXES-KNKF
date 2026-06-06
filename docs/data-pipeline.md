# KNK-VF Data Pipeline

The data pipeline follows the project tiers from `claude.md`:

- Tier 1: web-scale pretraining data, Russian corpora, code, STEM, technical docs
- Tier 2: instruction, reasoning, and preference data
- Tier 3: NULLXES domain data for banking, telecom, and HR
- Tier 4: safety, rejection, and red-team data

## Preparation Flow

```text
raw text -> NFKC normalize -> deduplicate -> quality filter -> PII scrub -> shard
```

The reference implementation is in `knk/data/pipeline.py`. It provides a local
JSONL preparation path for dry-runs. Production target training should convert
approved shards into Megatron `.bin` and `.idx` datasets.

## Tokenizer

`knk/tokenizer/train_tokenizer.py` trains a 128K BPE SentencePiece tokenizer with:

- NFKC normalization
- byte fallback
- `<|bos|>`, `<|eos|>`, and `<|pad|>` special tokens
- chat-template support for SFT

Use at least 10GB of representative EN/RU/code data for the first tokenizer
candidate and validate compression, unknown-token behavior, and language balance.

## Data Mix

The initial pretraining mix is configured in
`configs/data/pretrain_mix_tier1.yaml`:

- 45% English general web
- 15% Russian curated corpora
- 20% code and documentation
- 10% STEM and technical text
- 10% instruction and reasoning seed data

Treat these weights as the first proxy run defaults. Adjust only based on eval
results, loss curves, and language/domain regressions.

## Reproducibility

Every production shard should record:

- source
- license class
- quality filter version
- PII scrubber version
- tokenizer version
- data version hash
