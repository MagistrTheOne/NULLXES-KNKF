# KNKF Bootstrap Phase

Bootstrap artifacts prepared while random-weight initialization runs.

## Goals

- 128K multilingual tokenizer spec for EN + RU + code
- Chat template v2 with NULLXES identity
- Bootstrap identity / capability seed datasets
- HF tokenizer metadata export

## Tokenizer

Config: `configs/tokenizer/knk_vf_tokenizer_128k.yaml`

Special tokens:

- `<|bos|>`, `<|eos|>`, `<|pad|>`
- `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`
- `<|code|>`, `<|/code|>`, `<|think|>`, `<|/think|>`

Corpus mix target:

- 35% English web
- 25% Russian web
- 25% code
- 10% technical docs
- 5% NULLXES identity seed

## Chat Template

Use `knk/tokenizer/templates/knk_vf_chat_v2.jinja`.

Default system prompt identifies the model as KNK-VF by NULLXES and sets EN/RU/code enterprise behavior.

## Bootstrap Datasets

Tier 0 identity seed lives in `configs/data/bootstrap_identity_tier0.yaml` and `data/bootstrap/*.jsonl`.

These are seed files for:

- NULLXES identity EN/RU
- bilingual capability
- code capability
- enterprise safety

They are not sufficient for real training alone. Expand with proprietary NULLXES corpora before SFT.

## HF Export

Export tokenizer metadata bundle:

```bash
python -m knk.tokenizer.export_hf_tokenizer \
  --config configs/tokenizer/knk_vf_tokenizer_128k.yaml \
  --output /workspace/artifacts/knk_vf_tokenizer_hf
```

Copy the bundle into the model checkpoint directory before or after weight initialization upload.
