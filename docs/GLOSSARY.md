# Fine-tuning glossary

This reference assumes basic Python, not prior experience training language models.
Each notebook explains its own terms before the experiment; use this page to review
them. NovaBot is a **fictional teaching robot** with three modes: `idle`, `mapping`,
and `navigation`. Its examples explain training signals, not measured model results.

## Three different questions

| Dimension | Question | Examples |
|---|---|---|
| Training objective | What signal should improve the model? | Next-token prediction, demonstration imitation, preference optimization, reward maximization, distribution matching |
| Parameter update strategy | Which numbers can change, and how are they stored? | Full tuning, selective tuning, LoRA, QLoRA |
| Feedback source | Who or what supplies the target or judgment? | Documents, demonstrators, humans, model judges, programs, teachers |

These are not competing choices. Supervised Fine-Tuning (SFT) can use Low-Rank
Adaptation (LoRA). Reinforcement Learning from Human Feedback (RLHF) describes a
feedback source and workflow, not one mandatory optimizer. Proximal Policy
Optimization (PPO) is an optimization algorithm. Reinforcement Learning with
Verifiable Rewards (RLVR) is not synonymous with Group Relative Policy Optimization
(GRPO). See [the method comparison](TRAINING_METHODS.md).

## Models and parameters

See [00A: Qwen in PyTorch](../notebooks/00a_qwen_in_pytorch.ipynb).

| Term | Meaning and example |
|---|---|
| Large Language Model (LLM) | A model that predicts and generates language tokens using learned parameters. A text question can become a sequence of next-token predictions. |
| Vision-Language Model (VLM) | A model that combines image information with language. NovaBot's photographed status panel can be part of a question. |
| Pre-training | Initial broad training, often using next-token prediction over large collections. It establishes capabilities before a task-specific experiment. |
| Fine-tuning | Further training of an existing model for a domain, behavior, or task. It updates parameters; merely adding a manual to a prompt does not. |
| Parameter / weight | A learned numerical value, usually part of a tensor. “Weights” often refers collectively to all learned parameters, including more than matrix weights. |
| Configuration | The model's structural settings, such as layer count and hidden size. It is a blueprint, not the learned weights. |
| State dictionary | A mapping from parameter/buffer names to tensors. It holds numerical model state, unlike the structural configuration. |
| Forward pass | Computing outputs from inputs with the current parameters. For a language model, outputs include scores for possible next tokens. |
| Logit | An unnormalized token score. Softmax converts the scores into a probability distribution. |
| Gradient / backward pass | A gradient describes how loss changes with a parameter; backpropagation computes these derivatives. The optimizer subsequently updates trainable parameters. |
| Frozen parameter | A parameter excluded from updates. A frozen base model can still participate in the differentiable computation used to train an adapter. |
| Checkpoint | A saved state. An inference checkpoint may contain only model assets; exact training resume also needs optimizer, scheduler, random-number and progress state. |
| Full tuning | Updating all model parameters. It uses substantial memory for gradients and optimizer state. |
| Selective tuning | Updating a chosen subset, such as the last language layers. Verify parameter names rather than assuming a regex selects the intended modules. |
| Low-Rank Adaptation (LoRA) | Keeping base weights fixed while learning a small low-rank weight update. If the update is the product of two thin matrices, fewer values need training. |
| Quantized Low-Rank Adaptation (QLoRA) | LoRA with a quantized frozen base, typically stored at 4-bit precision in this project. Computation uses higher-precision values; this is not training every base weight in 4-bit precision. |
| Adapter / rank | An adapter stores added trainable components. LoRA rank sets the inner dimension of its low-rank update; larger rank increases adapter capacity and size. |

## Data and the training loop

See [00B: Local data to batches](../notebooks/00b_local_data.ipynb).

| Term | Meaning and example |
|---|---|
| Sample / record | One dataset item: a document, a conversation, or a preference pair. It may encode into more than one training sequence. |
| Token / tokenizer | A token is an element of the model's discrete vocabulary. A tokenizer maps text to token IDs and back; one word need not equal one token. |
| Chat template | Formatting rules for roles, separators and special tokens. The template tells the model where a user request ends and an assistant response begins. |
| Prompt / completion | The context supplied to generation / the continuation generated or provided as a target. A prompt can contain several conversation turns. |
| Batch / microbatch | A group processed together / one such group within gradient accumulation. A batch has a batch dimension as well as a sequence dimension. |
| Epoch | One pass over a finite training dataset. Streaming experiments often specify a step budget instead. |
| Step | Context-dependent: it may mean a batch forward/backward pass or an optimizer update. With four accumulated microbatches, four backward passes produce one optimizer step. |
| Gradient accumulation | Summing scaled gradients from multiple microbatches before updating parameters. Correctly handle the last incomplete accumulation group. |
| Label | A target used to compute an objective. Causal-language-model labels are token IDs; a preference label instead indicates which response was preferred. |
| Attention mask | Indicates which positions are valid context rather than padding. Causal masking separately prevents looking at future tokens. A visible prompt need not receive loss. |
| Loss mask / `-100` | Selects positions that contribute to the objective. In these causal-loss batches, a label of `-100` means “ignore this target,” not “remove this input.” |
| Causal shift | The logit at position `t` predicts the token at `t+1`. The model's built-in causal loss shifts internally; manual losses must align logits and labels once, not twice. |
| End-of-sequence (EOS) token | Marks a sequence/document boundary. In ordinary packed causal attention it does not itself block attention into a preceding document. |
| Packing / padding | Packing joins short sequences to use space efficiently; padding fills unused positions to make a rectangular batch. Padding must not contribute target loss. |
| Loss / objective | A quantity the optimizer tries to reduce. Its meaning depends on the training method; low loss alone does not establish useful behavior. |
| Optimizer / learning rate | The update rule / its step-size setting. AdamW also maintains running statistics that matter when resuming training. |
| Scheduler | Changes the learning rate as training progresses, usually at optimizer-update boundaries. |
| Data leakage | Information from validation/test data entering training or model selection inappropriately. Split related conversations/documents by source, not independently by chunk. |

## Learning from documents and demonstrations

See [01: DAPT](../notebooks/01_dapt.ipynb) and [02: SFT](../notebooks/02_sft.ipynb).

- **Continued Pre-training (CPT):** continuing a pre-training-style objective on an
  existing model. **Domain-Adaptive Pre-Training (DAPT)** is CPT focused on a domain,
  such as robot manuals. “NovaBot has three modes” supplies its own next-token
  targets: after “NovaBot has,” the next observed token is a target. “Unlabeled”
  means no separately written answers, not no supervision signal.
- **Supervised Fine-Tuning (SFT):** learning from desired outputs, or
  **demonstrations**. An **instruction** specifies what to do; an example pairs
  “How many modes does NovaBot have?” with “Three: idle, mapping, navigation.”
- **Teacher forcing:** predicting the next target token while conditioning on the
  preceding *provided target tokens*, not the model's earlier sampled guesses.
  This does not require a separate teacher model.
- **Assistant-only loss:** computing loss on assistant response targets while
  keeping the user prompt visible as context. The notebook verifies the actual
  role and padding masks; loss masking and attention masking are different.
- **Cross entropy / negative log-likelihood (NLL):** a penalty for assigning low
  probability to a target. For a constructed one-token example, probability 0.8
  gives `-ln(0.8) = 0.223`; probability 0.2 gives `1.609`. Natural logarithms are used.

Both DAPT and SFT can use the same causal token loss. Their records and supervised
positions differ. Reading manuals does not directly teach a question-answer format;
imitating an answer does not guarantee that it is true.

## Preferences, feedback, and policy optimization

See [03: DPO](../notebooks/03_dpo.ipynb) and
[04: Reward modeling and PPO](../notebooks/04_rlhf_rlaif.ipynb).

| Term | Meaning and example |
|---|---|
| Preference pair | One prompt with a preferred (`chosen`) and less preferred (`rejected`) response. “Three modes” versus “Five modes” is an instructional pair; preference is not automatically factual truth. |
| Policy | A model viewed as a probability distribution over possible outputs given a prompt. The language model generating an answer is the policy. |
| Reference policy | A frozen comparison model. It supplies baseline probabilities, not necessarily correct answers. |
| Log-probability | The logarithm of a probability. Summing conditional token log-probabilities gives a completion's log-probability; averaging instead changes the quantity. |
| Direct Preference Optimization (DPO) | Optimizing preferred-versus-rejected response odds relative to a frozen reference, directly from pairs, without first fitting a separate reward model. |
| Preference margin | A difference between chosen and rejected scores. In the DPO notebook, the log-probability margin is compared with the reference margin and scaled by `beta` before the logistic loss. |
| Reinforcement Learning from Human Feedback (RLHF) | A workflow using human judgments, often preferences to fit a reward model followed by policy optimization. It is not the name of one optimizer. |
| Reinforcement Learning from AI Feedback (RLAIF) | A workflow using model-generated judgments. Record the judge and raw decision so labels can be audited; AI judgments can be wrong or biased. |
| Reward model | A model producing a scalar score for a prompt and answer. Pairwise training teaches ordering, not a calibrated probability of truth. |
| Rollout / trajectory | A generated completion and the information collected around it, such as token probabilities and rewards. In language generation the actions are generated tokens. |
| Value estimate | A prediction of expected return. It provides a baseline for judging whether a sampled outcome was better than expected. |
| Return / advantage | Return is the reward signal attributed to a decision; advantage compares that signal with a baseline. Positive advantage encourages the sampled action under the objective. |
| Kullback-Leibler (KL) divergence | A directional comparison of probability distributions. Exact KL is nonnegative; a sampled single-response log-ratio used as an estimate can be negative. |
| Proximal Policy Optimization (PPO) | Policy optimization using a clipped probability-ratio objective to discourage overly large updates from the policy that collected the rollout. |
| Clipping | Limiting a ratio or another quantity to a range in an objective. PPO clipping is not the same as clipping gradient norms. |
| Generalized Advantage Estimation (GAE) | Estimating advantages by combining temporal-difference information across a trajectory. This course's scalar, sequence-level PPO approximation does not implement token-level GAE. |

**Implementation boundary:** the PPO lesson uses one sequence reward/value and
ratios derived from mean completion log-probabilities. These are not joint sequence
probability ratios or a complete token-level PPO algorithm. Read the explicit
equations and limitations in the notebook before transferring the idea to production.

## Verifiable rewards and group comparisons

See [05: GRPO / RLVR](../notebooks/05_grpo_rlvr.ipynb).

- **Reinforcement Learning with Verifiable Rewards (RLVR):** using a program to
  check outputs. A NovaBot verifier can award 1 for a parsed mode count of 3 and 0
  otherwise. Such a check verifies only what it encodes, not the entire answer.
- **Group Relative Policy Optimization (GRPO):** sampling several responses to
  the same prompt and using within-group rewards to derive relative advantages,
  rather than learning a separate value model in this lesson.
- **Reward variance / standard deviation:** measures how much rewards differ.
  Constructed rewards `[1, 0, 1, 0]` have mean 0.5 and population standard deviation
  0.5, yielding standardized advantages `[1, -1, 1, -1]` (ignoring a tiny stability
  constant). All-equal rewards yield zero centered advantages: no relative reward
  signal. A separate regularizer could still produce a gradient if enabled.
- **Reward hacking:** maximizing the implemented score without achieving the
  intended goal. A count-only verifier may reward a nonsensical answer containing
  the number 3; a length reward teaches length, not correctness.

The tiny notebook uses a length reward to expose mechanics on random outputs.
The NovaBot correctness example is explanatory, not a claim about those rollouts.

## Distillation

See [06: GKD](../notebooks/06_gkd.ipynb).

- **Knowledge distillation / Generalized Knowledge Distillation (GKD):** training
  a student to match information from a teacher. This lesson uses the student's
  own generated contexts and compares teacher/student next-token distributions.
- **Teacher / student:** the fixed model supplying target distributions / the
  trainable model learning from them. The teacher need not be perfect.
- **Hard / soft targets:** a single target token / a distribution across tokens.
  A constructed teacher distribution `[0.8, 0.2]` distinguishes plausible alternatives
  that the hard target “first token” alone would not describe.
- **Temperature:** a positive divisor applied to logits before softmax. Higher
  temperature makes a fixed nonuniform distribution softer. It is not a probability.
- **On-policy:** collecting contexts from the policy currently being trained.
  Here the student generates completions, and both models score those contexts.
- **Forward / reverse KL:** in this lesson, forward means teacher-to-student
  `KL(teacher || student)` and reverse means `KL(student || teacher)`. Always check
  argument order because terminology depends on the chosen convention.
- **Jensen-Shannon divergence (JSD):** compares distributions with a shared mixture.
  For the constructed teacher `[0.8, 0.2]` and student `[0.6, 0.4]`, forward KL is
  about 0.0915 and equally weighted JSD about 0.0242, using natural logs.

The notebook explains the locked trainer's piecewise `beta` convention: endpoints
select forward/reverse KL; interior values select a weighted JSD. It does not add
an automatic temperature-squared multiplier. Copying a differently normalized
distillation formula would change the experiment.

## Images and text

See [07: Image SFT](../notebooks/07_image_sft.ipynb).

| Term | Meaning and example |
|---|---|
| Multimodal | Using more than one input/output modality. Here an image and text jointly condition an answer. |
| Image patch | A small spatial part of an image represented in the vision computation. It is not a word token. |
| Vision encoder | Neural modules converting image pixels/patches into visual features. |
| Projector / merger | Components transforming and/or combining visual features for the language model. Inspect Qwen's actual merger instead of assuming every VLM has one generic linear projector. |
| Processor | Coordinated text tokenization and image preprocessing. It supplies token IDs, image tensors and geometry needed by the model. |
| Visual placeholder | A special position marking where image features enter the language sequence. It must align with processor output; it is not an answer token to imitate. |
| Visual loss mask | Excludes visual placeholder targets from the language loss. Answer loss can still backpropagate through trainable visual components that helped compute the answer. |

A fictional red NovaBot panel paired with “What color is the panel?” and “Red”
illustrates grounding: the answer should depend on the image. Correct-looking text
alone does not prove the model used the image. Never truncate away only part of the
visual representation to force a sequence to fit.

## Evaluation and saved artifacts

See [08: Evaluation and artifacts](../notebooks/08_evaluation_artifacts.ipynb).

| Term | Meaning and example |
|---|---|
| Baseline | The before-training result measured with the same preprocessing and evaluation settings as the trained model. |
| Train / validation / test | Data for parameter updates / development decisions / the final held-out report. Repeatedly choosing settings from test results turns the test into development data. |
| Generalization | Performing well beyond memorized training examples. Holding out a paraphrase of a seen manual fact tests something narrower than holding out a new manual. |
| Overfitting | Fitting training-specific patterns without corresponding held-out improvement. A decreasing training loss is not sufficient evidence against it. |
| Perplexity | The exponential of mean token NLL, using the same target mask. Loss `ln(2)` gives perplexity 2, not 50% answer accuracy. Compare only compatible tokenization and evaluation setups. |
| Greedy generation | Choosing the highest-probability next token at each generation step. It supports a controlled reload check; one matching output does not prove complete model equivalence. |
| Inference artifact | Model or adapter assets needed to load and generate, often saved with `save_pretrained()`. It need not contain training-state information. |
| Resume | Continuing training with restored weights, optimizer, scheduler, random-number state and progress. Loading weights alone starts a new optimizer trajectory. |
| Adapter dependency | A saved LoRA adapter needs its compatible base model and matching tokenizer/processor. It is not a self-contained full model. |
| Merge | Incorporating an adapter update into base weights for deployment. This course reloads an unquantized base before merging; keep the original adapter/base for reproducibility. |

## Quick self-check

1. Can an SFT experiment use LoRA and human-written demonstrations together?
2. Do all-equal GRPO rewards prove that every response is correct?
3. Does masking a user prompt out of loss remove it from the model's context?

<details>
<summary>Answers</summary>

1. Yes: these specify the objective, update strategy and feedback source, respectively.
2. No: every response might be wrong, or the verifier might be uninformative.
3. No: the prompt can remain visible through attention while its targets are ignored.

</details>
