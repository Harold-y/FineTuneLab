# Qwen3.5 2B and 4B Architecture

This document describes the dense Qwen3.5 checkpoints supported by FineTuneLab.
The source of truth is the official model configuration and model card, not the
summary here.

## High-level design

Qwen3.5 is a causal decoder with a native vision encoder. Text, image, and video
representations were trained in a unified system, although FineTuneLab v1 trains
only text and still-image inputs. The language stack alternates three Gated
DeltaNet linear-attention blocks with one gated full-attention block. This 3:1
pattern reduces the quadratic cost of attending over long visual and textual
contexts while retaining periodic global attention.

| Property | Qwen3.5-2B | Qwen3.5-4B |
| --- | ---: | ---: |
| Approximate language parameters | 2B | 4B |
| Hidden size | 2,048 | 2,560 |
| Language layers | 24 | 32 |
| Hybrid groups | 6 | 8 |
| Layers per group | 3 DeltaNet + 1 full attention | 3 DeltaNet + 1 full attention |
| DeltaNet V/QK heads | 16 / 16 | 32 / 16 |
| DeltaNet head dimension | 128 | 128 |
| Full-attention Q/KV heads | 8 / 2 | 16 / 4 |
| Full-attention head dimension | 256 | 256 |
| Rotary dimension | 64 | 64 |
| FFN intermediate size | 6,144 | 9,216 |
| Padded vocabulary | 248,320 | 248,320 |
| Native context | 262,144 | 262,144 |

## Gated DeltaNet and full attention

DeltaNet is a recurrent/linear-attention mechanism: it updates a compact state
instead of materializing a full token-by-token attention matrix. Gates decide how
strongly new information changes that state. Its practical value is lower memory
growth on long inputs. Periodic full-attention layers recover direct token-to-token
interaction that a compressed recurrent state may not preserve perfectly.

The full-attention layers use grouped-query attention (GQA): multiple query heads
share fewer key/value heads. This reduces KV-cache memory. Attention output gates
provide another learned control path around the attention result.

## Position, feed-forward, and output layers

Qwen3.5 uses multimodal rotary position information so text and spatial/temporal
visual positions can coexist. Each attention or DeltaNet operation is followed by
a SiLU-gated feed-forward network. The language-model output projection is tied to
the token embedding, so both references share weights.

The checkpoints were trained with multi-token prediction (MTP). MTP is a
pre-training technique and inference optimization signal; ordinary downstream
fine-tuning still uses the standard causal language-model loss unless a recipe
explicitly implements another objective.

## Vision path

The vision tower follows the Qwen3-VL encoder design. The processor converts an
image into pixel tensors and grid metadata, and the vision tower maps those inputs
to representations consumed by the language backbone. Image placeholder tokens
must stay aligned with these representations. Arbitrary sequence truncation can
remove placeholders and make a batch invalid, which is why FineTuneLab defaults
to `max_length: null` for image-text recipes.

## Base and post-trained checkpoints

- `Qwen3.5-2B-Base` and `Qwen3.5-4B-Base` contain pre-trained representations and
  are the recommended starting points for DAPT.
- `Qwen3.5-2B` and `Qwen3.5-4B` include post-training and are useful starting
  points for task SFT, preference alignment, and reasoning-focused reinforcement
  learning.
- A previous FineTuneLab stage may be used as the next stage's checkpoint. Record
  that lineage instead of silently returning to the original model.

Qwen3.5 thinks by default in its official chat template. Training data should be
consistent about whether reasoning traces are targets. Do not mix hidden/private
reasoning with short direct answers accidentally.

## What each tuning strategy changes

- **Full:** every language, vision, merger/projector, embedding, and output weight.
- **Selective:** the configured name patterns. The default selects the last four
  language layers plus final normalization/output modules and excludes vision.
- **LoRA:** low-rank updates on language linear layers. Vision is excluded by
  default; explicitly list merger/projector modules when adapting visual grounding.
- **QLoRA:** the same adapter objective while the frozen base is loaded in 4-bit
  NF4 with double quantization and BF16 arithmetic.

Always run `ftlab model inspect` and review the trainable count. Architecture names
can change between upstream library versions, and zero matched parameters is a
configuration error.

## Memory and context

Native 262K context does not imply that fine-tuning at 262K is economical.
Activation memory, optimizer state, multiple DPO forwards, GRPO generations, and
teacher/student distillation multiply the cost. Begin around 1K–4K tokens, measure
peak memory, then scale deliberately. Gradient checkpointing saves activation
memory at the cost of recomputation; LoRA saves optimizer and gradient memory;
QLoRA also reduces frozen-weight memory.

## References

- [Official Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)

