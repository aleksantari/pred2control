# pred2control

**pred2control** is an open-source *curriculum-driven research repository* for building intuition around modern **VLA-style** and **generalist robotic policies**.

The project progresses from **simple action-only sequence models** to **distributional, multimodal action generators**, mirroring the conceptual evolution seen in systems such as ACT, Diffusion Policy, and recent VLA architectures.

Rather than optimizing for task performance, this repository is designed to **isolate representational differences** between unimodal and multimodal policy classes under tightly controlled conditions.

---

## Project Status (Beta)

This repository represents an **active research scaffold**, not a finished benchmark.

At the current stage:
- Both **MotorGPT-chunk** and **MotorFlow-chunk** are fully implemented.
- Models are trained successfully on the same dataset using identical preprocessing, batching, and chunking.
- Training stability and convergence are verified via basic test RMSE.
- Multimodal behavior is **not yet evaluated beyond sanity checks**.

The present focus is **architectural validation and curriculum clarity**, not final performance claims.  
Behavioral and multimodality-focused evaluations are explicitly planned as the next phase.

---

## Motivation

Most introductions to VLAs begin with large multimodal models and complex datasets.  
This repository takes the opposite approach:

> **Start simple, control variables aggressively, and introduce complexity only when it is necessary.**

The central question driving this work is:

**When does a robot policy need to represent a *distribution* over actions instead of a single best prediction?**

To answer this, we:
- keep data and training pipelines fixed,
- vary only the *policy representation*,
- and evaluate where unimodal regression breaks down under ambiguity.

---

## How to Read This Repository

If you are skimming:
1. Read this README for scope and intent.
2. Look at `models/` to see the two policy heads being compared.
3. Check `data.py` to understand the shared batch construction.
4. Training scripts in `scripts/` are intentionally minimal and explicit.
5. The accompanying curriculum text explains *why* each modeling decision exists.

This structure is intentional: differences in behavior should be attributable to **model class**, not tooling.

---

## Curriculum Focus

The curriculum is structured around a progressive refinement of **System 1** (learned action distributions):

- Introduce **action chunking** for temporal stability and practical inference rates
- Begin with **unimodal, action-only autoregressive modeling**
- Upgrade the action head to a **distribution-learning formulation** (flow matching)
- Keep interfaces, datasets, and training loops consistent
- Compare behavior under identical conditions

System 2 (reasoning, planning, imagination) is intentionally deferred and treated as future work.

---

## Models (`models/`)

These models share the same dataset, batch sampler, normalization, and comparable training setups.

### `motor_gpt_chunk`
**Unimodal autoregressive action chunk generator**

- Predicts the next *chunk* of actions via direct regression
- Operates on action-only context (no observations or rewards)
- Deterministic baseline
- Highlights error compounding and mode collapse under ambiguous futures


---

### `motor_flow_chunk`
**Multimodal autoregressive action chunk generator**

- Uses conditional flow matching to learn a *distribution* over action chunks
- Samples actions via ODE integration at inference time
- Supports multiple plausible futures from the same prefix
- Abstracts the distributional action heads used in diffusion-style robot policies

Rather than predicting actions directly, this model learns a conditional velocity field that transports noise into valid action chunks.

---

## Model Architecture
(coming soon)


---

## Environment setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate torch-cu128
```

---

## Dataset & Data Preparation

The dataset is a **Hugging Face SO-101 robot arm collection** intentionally constructed to include:
- shared trajectory prefixes
- divergent suffixes
- controlled multimodality

This makes it suitable for diagnosing **mode collapse vs mode coverage**.

### Data Workflow

Before training, run:

1. `notebooks/dataset.ipynb`  
   Downloads the dataset from Hugging Face

2. `notebooks/preprocessing.ipynb`  
   - Normalizes joint-position actions  
   - Chunks episodes into fixed-length action sequences  

Only after preprocessing should training scripts be executed.

---

## Training (`scripts/`)

Training entry points for both model families live in `scripts/`.

Design principles:
- single-file runners
- explicit configs
- no hidden abstractions

The goal is to make architectural comparisons transparent and easy to modify for ablations.

---

## Initial Results (Sanity Checks)

Both models were trained on the same dataset with identical preprocessing and optimizer settings.

**Test RMSE on held-out episodes:**

| Model | Test RMSE | Training Steps |
|------|----------|----------------|
| MotorGPT (chunked) | **0.535** | 1,000 |
| MotorFlow (chunked) | **0.425** | 3,000 |

Notes:
- MotorFlow requires more steps due to the nature of flow-matching objectives
- RMSE measures average prediction accuracy, **not multimodal behavior**
- These results confirm **correctness, convergence, and stable training**

At this stage, RMSE should be interpreted strictly as a **sanity check**, not as evidence of multimodal superiority.

---

## Planned Next Evaluations

The next phase focuses on **behavioral evaluation**, not loss metrics, which may include (subject to change):

- Stochastic rollouts from identical prefixes
- Repeated sampling to probe mode diversity
- Trajectory overlays and variance analysis
- Comparison of rollout stability vs mode coverage
- Best-of-\(K\) and oracle-style diagnostics under ambiguous futures

These evaluations are intentionally deferred until model correctness and training stability are firmly established.

---

## Why This Work Is Useful

This repository isolates **policy representation** as a first-order research variable under controlled conditions.

It is intended to support:
- diagnostic studies of multimodality
- ablations on chunk length and conditioning
- comparisons between regression and generative policy heads
- extensions toward observation-conditioned and reasoning-augmented policies

The emphasis is on **understanding failure modes**, not leaderboard performance.

---

## License & Contribution

This repository is open-source and intended for educational and research use.

Contributions, critiques, and extensions are welcome.

If you use this code or ideas in academic work, please cite or acknowledge:

Aleks Santari, *pred2control: Curriculum-driven analysis of unimodal vs distributional robotic policies*.

## Contact

**Aleks Santari**  
LinkedIn: https://www.linkedin.com/in/aleksantari
