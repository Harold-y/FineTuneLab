# Evaluation

`ftlab evaluate --config experiment.yaml` selects the saved `output_dir/final`
artifact, including PEFT adapters. It fails when the artifact is missing.
Use `--checkpoint /path/to/base` explicitly for a baseline, or supply another full
checkpoint/adapter directory. Set `model.local_files_only: true` for offline runs.

`evaluation.max_samples` bounds evaluation records and `generate_samples` bounds
greedy sample generation. Samples contain only the prompt and generated answer;
the final reference assistant answer is excluded from the generation prompt.
Reward-model evaluation produces pairwise metrics rather than language generations.

Each evaluation writes a timestamped `evaluations/` directory containing its
resolved configuration, manifest, metrics and `generated_samples.json`. The
checkpoint identity is recorded and the original training config is preserved.
SFT/DAPT perplexity is derived from token loss, not from preference/RL losses.

[Lesson 08](../notebooks/08_evaluation_artifacts.ipynb) executes baseline-versus-
adapter evaluation, exact CPU resume and adapter merging end to end.

Always compare against the exact starting checkpoint and use a held-out split.
FineTuneLab records validation loss and method-specific trainer metrics. Useful
signals include DPO preference accuracy/margin, reward-model pairwise accuracy,
PPO reward and KL, GRPO reward variance and valid completion rate, and student
versus teacher divergence for GKD.

Loss alone is insufficient. Add task metrics, inspect generated samples, check
general capabilities for forgetting, and test both text and image inputs when a
vision component was updated. Keep decoding parameters identical across models.
