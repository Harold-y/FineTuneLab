# Training methods: choose a signal, not an acronym

Start with [the notebook course](../notebooks/README.md). Each chapter explains
terms, a worked objective, actual code variables, and misconceptions before its
first code cell. The [glossary](GLOSSARY.md) is a review aid, not a prerequisite.

## Three independent dimensions

| Dimension | Meaning | Example choice |
|---|---|---|
| Training objective | What should the model improve? | Imitate desired answers, prefer one response, maximize a reward, or match a teacher distribution |
| Parameter update strategy | Which parameters receive updates? | Full, selective, Low-Rank Adaptation (LoRA), or Quantized Low-Rank Adaptation (QLoRA) |
| Feedback source | Where does the target or judgment come from? | Existing text, demonstrations, human preferences, model judgments, or a programmatic verifier |

For example, **Supervised Fine-Tuning (SFT) with LoRA on human-written answers**
specifies all three dimensions. SFT and LoRA are not alternatives. Likewise,
**Reinforcement Learning from Human Feedback (RLHF)** and **Reinforcement Learning
from AI Feedback (RLAIF)** identify feedback sources/workflows, while **Proximal
Policy Optimization (PPO)** identifies an optimizer. RLHF does not mean PPO.
**Reinforcement Learning with Verifiable Rewards (RLVR)** identifies a kind of
reward; **Group Relative Policy Optimization (GRPO)** is one way to optimize with
such rewards. RLVR does not mean GRPO.

## One fictional manual, several training signals

NovaBot is an instructional robot with three modes: `idle`, `mapping`, and
`navigation`. The records below explain objectives. They are not additional
training data, measured outputs, or a promise of what a one-step experiment learns.
Runnable notebooks retain their small synthetic fixtures.

| Method | Example input and supervision | Desired change | What this alone cannot establish |
|---|---|---|---|
| Domain-Adaptive Pre-Training (DAPT) | Raw text: “NovaBot has three modes: idle, mapping, navigation.” Each observed next token is a target. | Better modeling of domain text | Reliable instruction following or factual recall |
| Supervised Fine-Tuning (SFT) | Question: “How many modes?” Demonstration: “Three: idle, mapping, navigation.” | Imitate the desired response given the question | Truth beyond the demonstrations |
| Direct Preference Optimization (DPO) | Same prompt, chosen “Three modes,” rejected “Five modes” | Improve the chosen/rejected odds relative to a frozen reference | Calibrated correctness probabilities |
| Reward model + PPO | A human or model prefers “Three”; a learned scorer then scores policy-generated responses | Increase learned reward while constraining policy changes | Immunity to scorer bias or reward hacking |
| GRPO / RLVR | Sample several counts; a program returns 1 for a parsed 3 and 0 otherwise | Encourage above-group-average verified responses | Correctness of explanations the verifier does not check |
| Generalized Knowledge Distillation (GKD) | Student-generated context; teacher gives next-token probabilities such as `[0.8, 0.2]` | Match a teacher distribution on student trajectories | Teacher correctness or capability equality |
| Image SFT | Image of a red status panel; question “What color?”; target “Red” | Learn an image-conditioned response | Proof that the model uses the image rather than text shortcuts |

## DAPT: domain text supplies its own targets

DAPT is **Continued Pre-training (CPT)** focused on a domain. After tokenizing the
manual, the logit at position `t` predicts the observed token at `t+1`. Thus raw
text needs no separately annotated question-answer labels. The objective is the
mean negative log-probability of valid next-token targets: assigning a constructed
target probability of 0.8 gives loss `-ln(0.8) ≈ 0.223`, whereas 0.2 gives `1.609`.

[Lesson 01](../notebooks/01_dapt.ipynb) exposes document boundaries, packing,
padding masks, and this causal shift. End-of-sequence markers do not automatically
isolate packed documents from one another's attention. A Base checkpoint is the
usual starting point. Domain adaptation can degrade previously learned abilities;
evaluate domain and general tasks rather than judging only training loss.

## SFT: demonstrations specify behavior

A demonstration pairs an instruction/prompt with a desired completion. During
**teacher forcing**, the model sees preceding target tokens while predicting the
next one; there is no separate teacher model. FineTuneLab applies the chat template
and normally computes loss on assistant/completion targets. User tokens remain
visible context even when their labels are `-100` (ignored by the loss).

[Lesson 02](../notebooks/02_sft.ipynb) connects raw question-answer records to role
masks and an explicit optimizer loop. SFT and DAPT may use the same causal loss;
the data organization and supervised positions distinguish the experiments.
For images, [lesson 07](../notebooks/07_image_sft.ipynb) also handles processor
outputs and visual placeholders. Masking visual targets does not prevent answer
gradients from reaching a trainable image merger.

## DPO: learn from comparisons

A preference pair contains `prompt`, `chosen`, and `rejected`. DPO compares the
policy's chosen-minus-rejected completion log-probability with the same margin
under a frozen reference policy. A logistic loss encourages an improved relative
margin; `beta` scales this reference-relative quantity. It is not a separately
trained reward model or an explicit PPO-style KL penalty.

[Lesson 03](../notebooks/03_dpo.ipynb) sums answer-token log-probabilities and checks
the objective numerically. Its held-out preference accuracy measures the sign of
the reference-relative margin, not answer factuality. Preference quality, response
length, and annotation ambiguity can all change what the model learns.

## Reward modeling, RLHF/RLAIF, and PPO

The human-versus-AI distinction concerns who labels the pair. A **reward model**
learns scalar scores whose ordering matches those labels. The **policy** then
generates a **rollout** (a sampled response), and the scorer assigns a reward. A
**value** estimates expected return; an **advantage** compares an outcome with a
baseline. A reference-policy **Kullback-Leibler (KL) divergence** penalty discourages
drift. PPO clips a policy probability ratio in its objective to discourage overly
large updates. None of these mechanisms guarantees factual answers.

**Important implementation limit:** [lesson 04](../notebooks/04_rlhf_rlaif.ipynb)
and the compact framework path teach a sequence-level PPO approximation. They use
scalar sequence values and ratios based on mean completion log-probabilities,
not standard token-level values/advantages or true joint-sequence ratios. They do
not implement token-level **Generalized Advantage Estimation (GAE)**. The notebook
explains what clipping and each diagnostic mean within this approximation.

This path is text-only. Learned rewards can be exploited, and sampled KL estimates
can be negative even though exact KL is nonnegative. Feedback artifacts preserve
judge identity, candidates, scores, decisions, raw output, and generation settings
for auditing. Model-generated judgments are not ground truth.

## GRPO / RLVR: compare sampled answers within a prompt

GRPO samples several completions for one prompt, scores them, and derives
advantages relative to their group. In the explicit lesson, constructed rewards
`[1, 0, 1, 0]` have mean 0.5 and population standard deviation 0.5, giving
advantages `[1, -1, 1, -1]` apart from a numerical stability constant. All-equal
rewards give no relative reward signal. A separate regularizer, if enabled, could
still update the model; the explicit notebook example uses zero KL coefficient.

[Lesson 05](../notebooks/05_grpo_rlvr.ipynb) uses a length reward on randomly
initialized model outputs to expose the mechanics. That is not a correctness
verifier. The NovaBot count checker illustrates RLVR, and must be replaced by an
appropriate verifier for real tasks. Monitor reward spread and inspect outputs:
high scores can mean that the model found a weakness in the rule.

## GKD: compare distributions on student-generated contexts

The student generates a continuation; a frozen teacher and trainable student
score next tokens on that same trajectory. These are **on-policy** contexts
because they come from the student being trained. **Soft targets** are probability
distributions rather than single token labels. **Temperature** divides logits
before softmax and changes the distribution's sharpness.

[Lesson 06](../notebooks/06_gkd.ipynb) defines each symbol and checks the loss against
the locked trainer. Its `beta` endpoints select forward/reverse KL, while interior
values use weighted **Jensen-Shannon divergence (JSD)**. The implementation does
not add a temperature-squared multiplier. All these details matter when comparing
formulas from other implementations. The teacher remains fixed; only student
parameters change. Real teacher/student runs are substantially more memory-intensive
than the tiny demonstration and do not guarantee a student as capable as its teacher.

## Parameter strategies can be combined with objectives

**Full** tuning updates all parameters. **Selective** tuning updates a named subset
that must be inspected. **LoRA** learns low-rank additions to frozen base weights.
**QLoRA** also quantizes the frozen base; it does not update every base weight in
4-bit precision. Multimodal adapter configurations can additionally train merger
components, so inspect the actual trainable-parameter report.

These strategies do not supply missing training data or verify target quality.
Not all implementation combinations are supported: in this version, quantized
reward-model/PPO stages are rejected. See [00A](../notebooks/00a_qwen_in_pytorch.ipynb)
for parameter and gradient inspection and
[08](../notebooks/08_evaluation_artifacts.ipynb) for adapter dependencies and merging.

## Choose stages from a measured need

A project might adapt to domain documents, then learn response format, then improve
preferences. Another may need only SFT; a verifiable task may benefit from GRPO;
distillation is a separate teacher-to-student option. These are choices, not a
mandatory pipeline. Establish a baseline, keep validation separate from final
test reporting, and add a stage only when its data and evaluation measure a clear
need. A successful one-step tiny run demonstrates computation, not model quality.
