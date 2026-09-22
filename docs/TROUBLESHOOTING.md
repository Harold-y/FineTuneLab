# Troubleshooting

## Out of memory

Reduce sequence/image size, batch size, and generation count; enable gradient
checkpointing; use LoRA/QLoRA; or move reference/teacher/rollout models to separate
GPUs. Clear evidence of the peak allocation is more useful than random changes.

## Loss is NaN

Check BF16 support, learning rate, empty completions, invalid labels, and reward
NaNs. Reproduce with eager attention and a one-sample batch before re-enabling
optimized kernels.

## Image-token mismatch

Set `max_length: null`, verify the image exists, and inspect the processor output.
Never truncate only `input_ids` after multimodal processing.

## No trainable parameters

Run `ftlab model inspect`. Selective patterns and upstream module names may not
match. FineTuneLab treats zero matches as an error.

## PPO instability

Inspect reward scale, KL, entropy, EOS rate, clipping fraction, and value loss.
Confirm that the reward model ranks held-out pairs correctly before policy updates.

