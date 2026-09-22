# Recipe Configuration

Every YAML file is a complete, reproducible experiment. The 2B directory contains
all seven training methods plus an image SFT and RLAIF-feedback example. The 4B
directory mirrors trainable 4B stages. Distillation intentionally has no 4B-student
mirror: the supported teaching experiment uses 4B as teacher and 2B as student.

Important fields:

- `model.name_or_path` selects one registered Qwen3.5 checkpoint or a local
  derivative.
- `data.modality` controls the capability matrix and image handling.
- `data.max_length: null` is the safe default for VLM data.
- `tuning.strategy` selects full, selective, LoRA, or QLoRA updates.
- `training.max_steps` overrides epochs and is useful for smoke tests.
- `model.reward_name_or_path` and `model.value_name_or_path` are required by PPO.
- `rewards` is an ordered, weighted list evaluated by GRPO.
- `judge` configures local or OpenAI-compatible RLAIF feedback generation.

The examples use tiny synthetic datasets and deliberately short runs. They are
starting points, not recommended hyperparameters for a real task.

