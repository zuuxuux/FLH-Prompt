# FLH-Prompt

# FLH-Prompts Implementation Plan

## Stage 1: Environment Setup

Create a Python environment with:
- torch, transformers, peft, datasets (HuggingFace ecosystem)
- wandb for experiment tracking
- numpy, scipy for JS divergence calculations

Verify GPU access. Target: single 8GB GPU sufficient for BERT-base.

---

## Stage 2: Data Pipeline

Load SST-2 from HuggingFace datasets. Create a streaming wrapper that:
- Yields batches continuously (cycle through dataset)
- Tracks global step count
- Accepts a `label_flip` flag that inverts labels (0↔1) when True
- Exposes a `flip_every_n` parameter (default 1000)

Test: stream 5000 samples, verify labels flip at correct intervals by printing samples at step 999, 1000, 1001.

---

## Stage 3: Frozen Backbone

Load `bert-base-uncased` with `AutoModelForSequenceClassification` (num_labels=2). Freeze all parameters. Create a helper function that:
- Takes input_ids, attention_mask, and a soft prompt tensor
- Prepends prompt embeddings to input embeddings
- Extends attention mask appropriately
- Returns logits

Test: random prompt tensor, random batch, verify forward pass produces logits of correct shape.

---

## Stage 4: FLHPromptPool Class

Create the core class with:

**State:**
- List of prompt tensors (nn.Parameter, each shape [prompt_length, embed_dim])
- List of weights (plain floats, not Parameters)
- Hyperparameters: alpha (temperature), prompt_length, embed_dim

**Methods:**

`birth_prompt()`: 
- Initialize new prompt (small random values, ~0.02 std)
- Apply mixing step: existing weights *= (1 - 1/t), new weight = 1/t

`update_weights(losses: list[float])`:
- Boltzmann update: w_i *= exp(-alpha * loss_i)
- Normalize so weights sum to 1

`get_prompt(mode='weighted_sum')`:
- 'weighted_sum': return weighted combination of all prompts
- 'sample': Thompson sample one prompt proportional to weights
- 'top_k': return highest-weight prompt

`get_all_losses(model, batch)`:
- Forward pass with each prompt
- Return list of cross-entropy losses
- Use torch.no_grad() for efficiency

Test: birth 3 prompts, synthetic losses [0.5, 1.0, 2.0], verify weights shift toward low-loss prompt after update.

---

## Stage 5: Training Loop

Single training step function that:
- Gets current prompt from pool (weighted_sum mode)
- Forward pass through frozen model
- Computes cross-entropy loss
- Backward pass (only prompt parameters have grad)
- Optimizer step (AdamW, lr ~1e-3)

Outer loop:
- Initialize pool with one prompt
- For each batch:
  - If step % birth_interval == 0: birth new prompt
  - Get losses for all prompts, update FLH weights
  - Train step on current prompt
  - Log: step, loss, accuracy, weight distribution, num_prompts
- Flip labels at specified intervals

Test: overfit on 10 batches (no flipping), verify loss decreases.

---

## Stage 6: Baselines

**Baseline 1 — Single Prompt:**
- One prompt, standard training loop
- No FLH machinery

**Baseline 2 — Fixed Pool + Random:**
- K prompts initialized at start
- Each batch, randomly select one to train
- No weight updates

**Baseline 3 — Fixed Pool + Input Similarity (mini-L2P):**
- K prompts, each with a learned key vector
- Selection: cosine similarity between input embedding (CLS token) and keys
- Train selected prompt + update its key

All baselines use same frozen backbone, same data stream, same flip schedule.

---

## Stage 7: Evaluation Metrics

Implement tracking for:
- `accuracy_over_time`: list of (step, accuracy) tuples
- `adaptive_regret`: mean accuracy over last N steps (e.g., N=500)
- `recovery_time`: steps after each flip to reach 80% accuracy
- `weight_entropy`: -sum(w * log(w)) — measures concentration of portfolio

Create plotting functions:
- Accuracy vs step (all methods on same plot)
- Weight distribution over time (heatmap: prompt index vs step)
- Recovery time bar chart per method

---

## Stage 8: First Experiment Run

Configuration:
- Dataset: SST-2
- Backbone: bert-base-uncased (frozen)
- Prompt length: 20 tokens
- FLH alpha: 0.1 (tune later)
- Birth interval: every 500 steps
- Flip interval: every 1000 steps
- Total steps: 10000 (10 regime flips)
- Batch size: 32

Run all four methods. Save:
- Checkpoints every 1000 steps
- Full metrics to wandb
- Final plots

---

## Stage 9: Ablations (after Stage 8 works)

**9a: Birth granularity**
- Birth every 250 / 500 / 1000 steps
- Adaptive: birth when entropy of predictions exceeds threshold

**9b: Alpha sensitivity**
- alpha in [0.01, 0.1, 0.5, 1.0]

**9c: Prompt length**
- 5, 10, 20, 50 tokens

**9d: Merging**
- Track loss history per prompt (last 100 batches)
- Compute pairwise JS divergence
- Merge prompts below threshold
- Compare: no merging vs parameter similarity vs behavioral (JS)

---

## Stage 10: Extension to Vision

Once text works, replicate with:
- Backbone: ViT-B/16 from timm or transformers (frozen)
- Dataset: CIFAR-10 with rotating labels
- Same FLHPromptPool class (just change embed_dim to 768)
- Same baselines

This validates cross-modality. Same algorithm, different domain.

---

## Directory Structure

```
flh-prompts/
├── data/
│   └── streaming.py          # Data pipeline with flip logic
├── models/
│   ├── frozen_backbone.py    # BERT/ViT wrapper with prompt injection
│   └── flh_pool.py           # FLHPromptPool class
├── baselines/
│   ├── single_prompt.py
│   ├── fixed_random.py
│   └── input_similarity.py
├── training/
│   └── trainer.py            # Main training loop
├── evaluation/
│   ├── metrics.py
│   └── plotting.py
├── experiments/
│   ├── rotating_sentiment.py # Stage 8 config
│   └── ablations.py          # Stage 9 configs
└── configs/
    └── default.yaml          # Hyperparameters
```

---

## Success Criteria for Stage 8

1. All four methods run without crashing for 10k steps
2. Single prompt shows sawtooth accuracy pattern (crash at flip, slow recovery)
3. Input similarity baseline fails to recover (accuracy stays low after first flip)
4. FLH shows fastest recovery after flips
5. Weight distribution plot shows mass shifting to recent prompts after each flip

If FLH doesn't beat baselines on this synthetic task, debug before scaling to harder benchmarks.

---

## Running Experiments

### Installation

```bash
git clone https://github.com/zuuxuux/FLH-Prompt.git
cd FLH-Prompt
uv sync
```

### Quick Start Examples

**Run FLH on Amazon multi-domain sentiment (recommended first experiment):**
```bash
uv run flh-prompt train \
    --method flh \
    --dataset amazon \
    --steps 10000 \
    --steps-per-domain 1000 \
    --birth-interval 500 \
    --alpha 0.1 \
    --run-name "flh_amazon"
```

**Run all 4 methods on Amazon for comparison:**
```bash
# FLH (our method)
uv run flh-prompt train --method flh --dataset amazon --steps 10000 --steps-per-domain 1000 --birth-interval 500 --alpha 0.1 --run-name "flh_amazon"

# Single prompt baseline
uv run flh-prompt train --method single --dataset amazon --steps 10000 --steps-per-domain 1000 --run-name "single_amazon"

# Random selection baseline
uv run flh-prompt train --method random --dataset amazon --steps 10000 --steps-per-domain 1000 --run-name "random_amazon"

# Input similarity baseline (L2P-style)
uv run flh-prompt train --method similarity --dataset amazon --steps 10000 --steps-per-domain 1000 --run-name "similarity_amazon"
```

**Run FLH on CIFAR-10 vision task:**
```bash
uv run flh-vision train \
    --method flh \
    --steps 10000 \
    --rotate-interval 1000 \
    --birth-interval 500 \
    --alpha 0.1 \
    --run-name "flh_vision"
```

**Run all 4 methods on CIFAR-10:**
```bash
# FLH
uv run flh-vision train --method flh --steps 10000 --rotate-interval 1000 --birth-interval 500 --alpha 0.1 --run-name "flh_vision"

# Single prompt
uv run flh-vision train --method single --steps 10000 --rotate-interval 1000 --run-name "single_vision"

# Random selection
uv run flh-vision train --method random --steps 10000 --rotate-interval 1000 --num-prompts 10 --run-name "random_vision"

# Input similarity
uv run flh-vision train --method similarity --steps 10000 --rotate-interval 1000 --num-prompts 10 --run-name "similarity_vision"
```

**SST-2 with label flipping:**
```bash
uv run flh-prompt train --method flh --dataset sst2 --steps 10000 --flip-interval 1000 --run-name "flh_sst2"
```

### CLI Options

```
Text CLI (flh-prompt):
  --method          flh | single | random | similarity
  --dataset         sst2 | amazon
  --steps           Total training steps (default: 10000)
  --flip-interval   Steps between label flips for SST-2 (default: 1000)
  --steps-per-domain Steps per domain for Amazon (default: 1000)
  --birth-interval  Steps between prompt births for FLH (default: 500)
  --alpha           FLH temperature (default: 0.1)
  --lr              Learning rate (default: 0.001)
  --batch-size      Batch size (default: 32)
  --run-name        Wandb run name

Vision CLI (flh-vision):
  --method          flh | single | random | similarity
  --steps           Total training steps (default: 10000)
  --rotate-interval Steps between label rotations (default: 1000)
  --birth-interval  Steps between prompt births for FLH (default: 500)
  --alpha           FLH temperature (default: 0.1)
  --num-prompts     Pool size for random/similarity (default: 10)
  --lr              Learning rate (default: 0.001)
  --batch-size      Batch size (default: 32)
  --run-name        Wandb run name
```

### Monitoring

Experiments log to [Weights & Biases](https://wandb.ai). View runs at your wandb project dashboard.

### Hardware

- Text (BERT): ~2GB GPU per run
- Vision (ViT): ~4GB GPU per run
- Can run 4-6 experiments in parallel on 24GB GPU