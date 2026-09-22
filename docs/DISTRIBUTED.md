# Distributed Training

FineTuneLab delegates distributed execution to Accelerate. Start with data
parallelism for LoRA and use DeepSpeed ZeRO-3 or FSDP when full model, optimizer,
and gradient states do not fit on one device.

```bash
accelerate config
accelerate launch -m finetunelab train -c configs/qwen35/4b/sft_lora.yaml
```

The repository includes a conservative ZeRO-3 example under `configs/accelerate`.
Do not combine `device_map="auto"` model sharding with a trainer-managed distributed
strategy. GRPO, PPO, and GKD also need memory for generation, reference, reward,
value, or teacher models; they often require separate devices even when SFT fits.

