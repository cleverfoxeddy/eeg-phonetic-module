# Kara-One Subject-Independent Phonological Decoding

## Overview

This project investigates **subject-independent phonological decoding** on the **Kara-One imagined speech EEG dataset**.  
The main goal was to compare whether a **learned latent manifold model** or a **simple fixed-feature + PCA + XGBoost pipeline** is more robust under a **small-sample LOSO setting**.

Originally, the learned manifold model was intended to be the main approach.  
However, a lightweight baseline using **fixed CNN+LSTM features + PCA + XGBoost** unexpectedly outperformed it, which led to a broader empirical comparison study on:

- representation complexity
- channel reduction
- compression loss
- robustness under cross-subject EEG variability

---

## Research Questions

1. In Kara-One subject-independent LOSO evaluation, which is more robust:
   - a learned latent manifold model (**P1+DAE**), or
   - a fixed-feature PCA baseline?

2. How does **channel reduction (64 channels vs paper10)** affect performance depending on model class?

3. Does PCA act as a simple dimensionality reduction step, or does it also provide **denoising / regularization**?

---

## Dataset

- **Dataset**: Kara-One
- **Setting**: subject-independent LOSO
- **Subjects**: 14
- **Tasks**: 5 binary phonological classification tasks
  - vowel
  - nasal
  - bilabial
  - iy
  - uw

---

## Compared Pipelines

### 1. P1+DAE
A learned latent manifold pipeline:
- SpatialCNN + TemporalLSTM
- 1152-dimensional merged feature
- encoder/decoder-based latent representation
- XGBoost classifier

### 2. PCA baseline
A simple fixed-feature pipeline:
- fixed random SpatialCNN + TemporalLSTM
- 1152-dimensional merged feature
- PCA to 32 dimensions
- XGBoost classifier

### 3. Upper-bound comparison
To estimate compression cost:
- **A.** `1152 → XGB`
- **B.** `1152 → PCA(32) → XGB`

---

## Main Results

### Overall accuracy

| Condition | Channels | Mean Accuracy |
|---|---:|---:|
| P1+DAE | 64 | 0.6298 |
| P1+DAE | 10 | 0.6173 |
| PCA baseline | 64 | 0.6444 ~ 0.6457 |
| PCA baseline | 10 | 0.6446 ~ 0.6461 |

### Upper-bound comparison

#### 64 channels
- `1152 → XGB`: **0.6491**
- `1152 → PCA32 → XGB`: **0.6431**
- Gap: **+0.0060**
- PCA explained variance: **0.7864**

#### paper10
- `1152 → XGB`: **0.6419**
- `1152 → PCA32 → XGB`: **0.6461**
- Gap: **-0.0042**
- PCA explained variance: **0.8984**

---

## Key Findings

1. **Simple pipelines can outperform learned latent manifolds**  
   Under a small-sample subject-independent EEG regime, fixed features + PCA + XGBoost were more robust than the learned P1+DAE model.

2. **The effect of channel reduction is model-dependent**  
   - P1+DAE benefited from 64 channels.
   - PCA baseline performed similarly or slightly better with paper10.

3. **PCA may act as denoising/regularization in low-channel settings**  
   In the paper10 condition, PCA(32) slightly outperformed the unreduced 1152-dimensional upper bound.

---

## Interpretation

These results suggest that:

- additional channel information is not always beneficial,
- low-channel subsets may become more compression-friendly,
- and in fixed-feature pipelines, PCA can remove noisy or redundant subspaces.

In short, **paper10 + PCA32 is not just a low-dimensional approximation, but a potentially more robust representation in this setting**.

---

## Limitations

- Only one dataset (**Kara-One**) was used.
- No downstream validation such as EEG-to-Text or LLM integration was performed.
- The contribution is primarily **empirical comparison**, not a novel model proposal.
- Generalization to other imagined speech datasets remains untested.

---

## Project Positioning

This is best understood as an **empirical comparison study** rather than a full research paper proposing a new state-of-the-art model.

The main contribution is:

- showing that **representation complexity matters**,
- demonstrating that **channel reduction effects depend on model class**,
- and suggesting that **simple, strongly regularized pipelines can be surprisingly robust** in small-sample cross-subject imagined speech EEG.

---

## Possible Future Directions

- SHAP-based channel importance analysis
- `paper10 + a` channel subset search
- validation on additional imagined speech datasets
- exploratory transfer experiments to reading EEG datasets such as ZuCo
- integration as a phonological submodule for EEG-to-Text systems

---

## Author Note

This project was originally conducted as a personal research exploration.  
Although not developed into a formal paper, it produced an interpretable empirical pattern that may be useful for future work on robust EEG representation design.
