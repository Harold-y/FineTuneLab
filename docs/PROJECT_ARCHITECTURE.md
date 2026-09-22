# Project Architecture

Start with the [execution/data flow diagrams](EXECUTION_AND_DATA_FLOW.md) and
[hands-on course](../notebooks/README.md). The course exposes the calculations
that the production modules below organize.

FineTuneLab follows dependency inversion: training recipes depend on small public
protocols, while Qwen- and library-specific details stay at the edges.

```text
CLI / workflows
    ├── typed configuration and capability validation
    ├── dataset loader and canonical schema validation
    ├── model-adapter registry
    ├── tuning strategy (full / selective / LoRA / QLoRA)
    ├── recipe factory (DAPT / SFT / DPO / RM / PPO / GRPO / GKD)
    └── manifests, metrics, checkpoints, evaluation, and export
```

## Module responsibilities

- `config.py` is the public experiment schema. Cross-field constraints belong
  here so invalid jobs fail before model allocation.
- `models/` owns architecture-specific loading and component discovery. A future
  model family should add an adapter rather than branching throughout recipes.
- `data/` converts local or Hub sources to canonical rows and resolves images.
- `tuning.py` is the only module that freezes parameters or creates PEFT models.
- `recipes/` translates stable FineTuneLab settings into version-specific trainer
  objects.
- `ppo.py` implements the text-only educational PPO loop because the locked TRL
  release has no PPO trainer.
- `rewards/` and `judges.py` separate verifiable rewards from model-based feedback.
- `workflows.py` owns run lifecycle and is shared by the CLI and notebooks.
- `runtime.py` owns seeding, environment reports, and reproducibility manifests.

## Adding a model family

Implement `ModelAdapter`, register it, add a capability check, and provide tests
for component discovery and all supported tuning strategies. Recipes should not
need changes unless the model requires a genuinely different objective.

## Adding a training method

Add a discriminated Pydantic recipe class, its canonical dataset alternatives, a
trainer factory, method-specific metrics documentation, and an offline one-step
test. Do not hide unsupported modality or quantization combinations behind runtime
fallbacks.

## Failure boundaries

Configuration and sampled dataset validation occur before model loading. Optional
dependencies produce actionable errors. A failed training run updates its manifest
without serializing secrets or raw API credentials.
