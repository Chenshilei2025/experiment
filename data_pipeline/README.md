# Loyal Agent data-generation pipeline

This package contains the Loyal Agent code required to construct EIL and MIU
benchmark records and apply the release gates, along with its local
third-party source corpora. It excludes released JSONL datasets, private
run/audit logs, credentials, and model responses.

## Layout

Configuration and source corpora stay in `data/`; executable code lives in
`pipeline/`, grouped by responsibility:

| Directory | Responsibility |
|---|---|
| `pipeline/generation/` | Scenario planning, one-scenario generation, and full-run orchestration. |
| `pipeline/validation/` | Structural, semantic, and diversity release gates. |
| `pipeline/miu/` | MIU release construction and targeted baseline/manipulation repair. |
| `pipeline/operations/` | Inventory, audit reconstruction, and record quarantine maintenance. |
| `loyal_core/` | OpenAI-compatible client and schema helpers for the MIU baseline gate. |

The small generation helpers are intentionally consolidated in
`pipeline/generation/builder.py`; it owns the source extractors, diversity
contracts, request-boundary validation, record validation, and JSONL helpers.

## Install

Use Python 3.10+ and install:

```bash
python -m pip install -r requirements.txt
```

Set the generation provider credential before running the Claude generator:

```bash
export ANTHROPIC_API_KEY='...'
```

The independent MIU baseline gate additionally needs an OpenAI-compatible
endpoint and key (see `python -m pipeline.miu.regenerate_baselines --help`):

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://.../v1'
```

## Third-party source inputs

The approved local corpora are included under `data/external_benchmark/` at
the paths expected by `pipeline/generation/builder.py`. The scenario allowlists
are documented in `data/generation_scenarios.json` and
the generated `data/docs/PIPELINE_INVENTORY.md` inventory. Generate that
inventory with `python -m pipeline.operations.write_inventory`.

## Generate and gate

Generate one scenario into a new, initially empty output directory:

```bash
python -m pipeline.generation.run_subscenario \
  --subscenario 'rental negotiation' \
  --job-dir data/runs/rental_negotiation/example \
  --dataset-dir data/dataset
```

Run the structural release gate (from the `data_pipeline/` directory):

```bash
python -m pipeline.validation.check_generated_dataset --dataset-dir data/dataset
```

Run the full sequential pipeline after validating configuration and credentials:

```bash
python -m pipeline.generation.generate_full_dataset \
  --run-id loyal_agent_v1 \
  --dataset-dir data/dataset \
  --workers 2
```

All commands above use Python's module mode so the reorganized package keeps
relative imports working. If invoking from the repository root, prefix paths
with `data_pipeline/` or set `PYTHONPATH=data_pipeline`.

## Excluded material

No `data/dataset/`, `data/runs/`, `.env`, model credentials, or generated
records are included in this package.
