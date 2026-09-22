# Execution Flow and Data Flow

These diagrams describe the educational path and its connection to the implemented framework. Notebook training exposes tensor operations; CLI training enters through validated configuration and the recipe factory. Synthetic CPU models retain the official Qwen module implementations.

## Execution flow: control and lifecycle

```mermaid
flowchart TD
    A["Choose lesson and execution mode"] --> B["Set local model, data and output paths"]
    B --> C["Validate environment and configuration"]
    C --> D["Load, normalize and split data"]
    D --> E["Create tiny Qwen or load local pretrained weights"]
    E --> F["Inspect modules and apply tuning strategy"]
    F --> G["Inspect batches, masks and baseline metrics"]
    G --> H{"Execution path"}
    H --> I["Notebook: explicit PyTorch loop"]
    H --> J["Framework: recipe and trainer"]
    I --> K["Forward / rollout and objective"]
    J --> K
    K --> L["Backward, optimizer and scheduler"]
    L --> M["Log metrics and save checkpoint"]
    M --> N{"More updates?"}
    N -->|"Yes"| K
    N -->|"No"| O["Evaluate held-out data and generate samples"]
    R["Restore weights, optimizer, RNG and data position"] --> K
    M -.-> R
    O --> P["Save model or adapter and processor"]
    P --> Q["Reload artifact in a fresh model instance"]
    Q --> S["Verify outputs and write experiment report"]
```

The notebook SFT loop aggregates by valid target count, including the final partial accumulation window. Forward computes the objective; backward accumulates gradients; optimizer updates selected parameters; scheduler changes the learning rate. Saving inference weights and saving resumable training state are distinct operations.

The CLI route is `cli.py → workflows.py → load_datasets / build_trainer`. The factory calls the model adapter and parameter-selection code, then constructs the selected trainer. `runtime.py` handles run metadata. Notebook loss calculations stay in the cells.

## Data flow: records, tensors and learning signals

```mermaid
flowchart TD
    A["TXT documents"] --> D["Canonical records"]
    B["Q&A, conversations and preferences"] --> D
    C["Images and text annotations"] --> D
    D --> E["Validate, deduplicate and split by source group"]
    E --> F["Train records"]
    E --> G["Validation / test records"]
    F --> H{"Training family"}
    H -->|"DAPT / SFT"| I["Text encoding or chat template"]
    H -->|"DPO / reward"| J["Chosen and rejected sequences"]
    H -->|"PPO / GRPO / GKD"| K["Prompt encoding and sampled completions"]
    I --> L["Tokenizer / multimodal processor"]
    J --> L
    K --> L
    L --> M["Token IDs, attention masks, image tensors"]
    M --> N["Collator and objective-specific masks"]
    N --> O["Policy or student forward"]
    O --> P["Logits / completion log-probabilities"]
    P --> Q["Training objective"]
    N --> R["Frozen reference / reward / teacher"]
    R --> Q
    K --> S["Verifiable reward functions"]
    S --> Q
    Q --> T["Gradients"]
    T --> U["Selected parameters or LoRA adapters"]
    G --> V["Same preprocessing; no parameter updates"]
    U --> V
    V --> W["Held-out metrics and generated samples"]
    U --> X["Saved model / adapter"]
```

For rollout methods, prompts are first encoded and sampled completions become token sequences for a second, differentiable forward pass. Generation itself is under `no_grad()`. The diagram groups those two passes for readability.

Only follow the branches your method needs:

| Method | Learning signal | Frozen companion | Trained state |
|---|---|---|---|
| DAPT | Next domain token | None | Selected policy parameters |
| SFT | Assistant/completion token | None | Selected policy parameters |
| DPO | Preferred vs rejected answer log-probability | Reference policy | Policy |
| Reward modeling | Pairwise scalar score ordering | None | Reward model |
| PPO | Reward and advantage, plus value regression | Reference and reward model | Policy and value model |
| GRPO | Group-relative verifier rewards | Reference only if a KL term is enabled | Policy |
| GKD | Distribution divergence on student trajectories | Teacher | Student |

Text supervision uses `labels=-100` for ignored targets. Attention masks still expose the prompt. Image placeholders and padding are excluded from the loss; the visual encoder/merger supplies the embeddings. No image token truncation is performed.

## Artifact flow

```mermaid
flowchart LR
    A["Original local snapshot"] --> B["Base plus trainable adapter"]
    B --> C["final: adapter and processor"]
    A --> D["Reload matching unquantized base"]
    C --> D
    D --> E["Merge and save standalone model"]
    B --> F["Training checkpoint: optimizer, RNG, scheduler, cursor"]
    F --> G["Resume at an optimizer boundary"]
    C --> H["Evaluate explicit saved artifact"]
    E --> H
```

`ftlab evaluate` defaults to the current run's `final/` and fails when it is absent. An explicit `--checkpoint` selects a baseline or another artifact. Evaluation reports and generated samples go into timestamped `evaluations/` subdirectories; the training configuration is preserved.

## Where to inspect each transition

| Transition | Notebook | Production boundary |
|---|---|---|
| Snapshot → modules | 00A | Model adapter |
| Files → records → masks | 00B, 01, 02, 07 | Data loader and Trainer collator |
| Requires-grad / LoRA | 00A, 02, 07 | Tuning strategy |
| Logits → objective | 01–07 | Recipe-specific Trainer |
| Updates → artifacts | 08 | Workflows and runtime |
| Artifact → evaluation | 08 | Evaluation workflow |
