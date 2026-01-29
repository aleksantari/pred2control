# pred2control

**pred2control** is an open-source *curriculum-driven research repository* for building intuition around modern **VLA-style** and **generalist robotic policies**.

The project progresses from **simple action-only sequence models** to **distributional, multimodal action generators**, mirroring the conceptual evolution seen in systems like ACT, Diffusion Policy, and recent VLA architectures.

Rather than optimizing for task performance, this repo is designed to **isolate representational differences** between unimodal and multimodal policy classes under controlled conditions.

---

## Motivation

Most introductions to VLAs begin with large multimodal models and complex datasets.  
This repo takes the opposite approach:

> *Start simple, control variables aggressively, and introduce complexity only when it is necessary.*

The central question driving this work is:

**When does a robot policy need to represent a *distribution* over actions instead of a single best prediction?**

---

## Curriculum focus

The curriculum is intentionally staged:

- Begin with **unimodal, action-only autoregressive modeling**
- Introduce **action chunking** to stabilize short-horizon rollouts
- Upgrade the action head to a **distribution-learning formulation** (flow matching)
- Keep data interfaces, optimizers, and training loops as consistent as possible
- Compare model behavior under identical datasets and evaluation setups

This structure allows differences in performance to be attributed primarily to **model class**, not tooling or training noise.

---

## Models (`models/`)

### `motor_gpt_chunk`
**Unimodal autoregressive action chunk generator**

- Predicts the next chunk of actions via direct regression
- Trained purely on action sequences (no observations or rewards)
- Serves as a deterministic baseline
- Highlights error compounding and mode collapse under ambiguous futures

### `motor_flow_chunk`
**Multimodal autoregressive action chunk generator**

- Uses conditional flow matching to learn a *distribution* over action chunks
- Samples actions via ODE integration at inference time
- Designed to represent multiple valid futures from the same prefix
- Mirrors the distributional action heads used in modern diffusion-style robot policies

---

## Dataset & data preparation

The dataset is a **Hugging Face SO-101 robot arm collection** intentionally constructed to include:
- shared trajectory prefixes
- divergent suffixes
- controlled multimodality

This makes it suitable for diagnosing **mode collapse vs mode coverage**.

### Data workflow
Before training, run:

1. `notebooks/dataset.ipynb`  
   Downloads the dataset from Hugging Face

2. `notebooks/preprocessing.ipynb`  
   - Normalizes actions  
   - Chunks episodes into fixed-length action sequences  

Only after preprocessing should the training scripts be executed.

---

## Training (`scripts/`)

Training entry points for both model families live in `scripts/`.

The scripts are intentionally minimal:
- single-file runners
- explicit configs
- no hidden magic

This makes them easy to modify for curriculum experiments and ablations.

---

## Initial results (current)

Both models were trained on the same dataset using identical normalization, chunking, and optimizer settings.

**Test RMSE on held-out episodes:**

| Model | Test RMSE | Training steps |
|------|----------|----------------|
| MotorGPT (chunked) | **0.535** | 1,000 |
| MotorFlow (chunked) | **0.425** | 3,000 |

Notes:
- MotorFlow requires more steps due to the nature of flow-matching objectives
- RMSE reflects *average prediction accuracy*, not multimodal behavior
- These results confirm **successful convergence and stable training** for both model classes

At this stage, RMSE should be interpreted as a **sanity check**, not as evidence of multimodal superiority.

---

## Planned next evaluations

The next phase of the curriculum focuses on **behavioral evaluation**, not loss metrics:

- Stochastic rollouts from identical prefixes
- Repeated sampling to probe mode diversity
- Trajectory overlays and variance analysis
- Comparison of rollout stability vs mode coverage

These evaluations are intentionally deferred until after model correctness and training stability are established.

---

## Environment setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate torch-cu128
