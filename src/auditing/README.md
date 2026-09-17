# Auditing Module

Black-box auditors that produce empirical **lower bounds** ε_audit on the
privacy of a DP-SGD trained model.  These lower bounds complement the
**upper bounds** computed by the accounting module, letting us measure
the *tightness gap* of differential privacy guarantees.

---

## One-Run Canary Auditing (`one_run.py`)

**Reference:** Steinke, Nasr & Jagielski, _"Privacy Auditing with One (1) Training Run"_, NeurIPS 2023. [arXiv:2305.08846](https://arxiv.org/abs/2305.08846)

### Algorithm (Steinke et al. 2023, §3–4)

The auditor participates **before, during, and after** training:

#### 1. Canary Construction (before training)

We follow Section 4 of the paper: canaries are **mislabeled copies of real
training examples**.

```
for i = 1, …, m:
    1. Sample a real example (xᵢ, yᵢ) uniformly from the training set.
    2. Draw ỹᵢ uniformly from {0, …, K−1} \ {yᵢ}   (random wrong label).
    3. The canary is (xᵢ, ỹᵢ).
```

**Why mislabeled real examples instead of random noise?**

| Strategy | On manifold? | Model can fit? | IN/OUT signal? |
|---|---|---|---|
| Random Gaussian noise | ✗ | ✗ — loss is equally high for IN and OUT | ✗ — ε_audit = 0 |
| Mislabeled real images | ✓ | ✓ — model tries to memorize the wrong label | ✓ — lower loss on ỹ when IN |

Random noise images lie far from the data manifold.  The model cannot
learn to fit them regardless of whether they are in the training set, so
the IN/OUT scores are indistinguishable and v ≈ r/2 → ε_audit = 0.

Mislabeled real examples sit on the manifold but conflict with the true
label.  When included in training, the model is incentivized to memorize
the wrong label (reducing its loss), producing a detectable IN signal.
When excluded, the model has no reason to predict the wrong label.

#### 2. Coin-Flip Inclusion (before training)

For each canary i, flip an independent fair coin:

```
bᵢ ~ Bernoulli(1/2)
```

- If `bᵢ = 1` → canary i is **IN** (added to the training set).
- If `bᵢ = 0` → canary i is **OUT** (withheld).

The training set becomes `D ∪ {(xᵢ, ỹᵢ) : bᵢ = 1}`.

#### 3. Training

Train the model with DP-SGD on the augmented training set.  The auditor
**does not** interact with training; it only provides the augmented dataset.

#### 4. Scoring (after training)

For each canary i, compute:

```
scoreᵢ = −CrossEntropyLoss(model(xᵢ), ỹᵢ)
```

- **Included canary** (bᵢ = 1): the model trained on (xᵢ, ỹᵢ), so its
  loss on the wrong label ỹ should be lower → score is higher (closer to 0).
- **Excluded canary** (bᵢ = 0): the model never saw label ỹ for input x,
  so its loss is higher → score is lower (more negative).

#### 5. Guessing

For each canary, the auditor guesses whether it was IN or OUT:

```
if   scoreᵢ > threshold:     guess IN  (+1)
elif scoreᵢ < −threshold:    guess OUT (−1)
else:                          abstain   (0)
```

With `threshold = 0` (default), we use a **median split**: guess IN for
above-median scores, OUT for below-median.

#### 6. Calibration (v, r) → ε_audit

Let:
- `r` = number of non-abstained guesses
- `v` = number of correct guesses among those r

Under (ε, δ)-DP, the number of correct guesses V is stochastically
dominated by `Bin(r, p(ε)) + δ-correction`, where:

```
p(ε) = eᵋ / (1 + eᵋ) = sigmoid(ε)
```

We compute **ε_audit** = largest ε such that the observed v is "too
surprising" at confidence 1−β:

```
Find ε* satisfying:  Pr[Bin(r, p(ε*)) ≥ v] + r·δ = β
```

All ε < ε* are ruled out: if the true privacy were smaller, the observed
success rate would occur with probability < β (a confidence-calibrated
lower bound).

The inversion is done by bisection (Brent's method) on the exact binomial
tail (`scipy.stats.binom.sf`).  See `calibration.py` for details.

### Key Parameters

| Parameter | Config Key | Default | Description |
|---|---|---|---|
| m (num canaries) | `auditing.one_run.num_canaries` | 500 | More canaries → tighter bound but slower training |
| threshold | `auditing.one_run.abstention_threshold` | 0.0 | Score threshold for abstention (0 = no abstention) |
| β (confidence) | `auditing.one_run.confidence_beta` | 0.05 | Bound holds w.p. ≥ 1−β |
| δ | `training.delta` | 1e-5 | The δ in (ε, δ)-DP |

### File Layout

```
src/auditing/
├── __init__.py           # Public exports
├── README.md             # This file
├── calibration.py        # (v, r) → ε_audit via binomial tail inversion
├── one_run.py            # One-run canary auditor (Steinke et al. 2023)
├── worst_case_init.py    # Worst-case init auditor (Annamalai 2024) [stub]
├── poisoning.py          # Poisoning-based auditor (Jagielski 2020)  [stub]
└── dpsniper.py           # DP-Sniper (Bichsel et al. 2021)          [stub]
```

---

## Calibration Module (`calibration.py`)

Converts audit outcomes (v, r) into a calibrated ε lower bound at
confidence level 1−β.  This is the primary theory deliverable for
**Problem II** in the project proposal.

### Methods

1. **Exact binomial inversion** (`method="exact_binomial"`):
   Bisection on `Pr[Bin(r, p(ε)) ≥ v] + r·δ = β` using `scipy.stats.binom.sf`.

2. **Chernoff approximation** (`method="chernoff"`):
   Closed-form Hoeffding-style bound. Faster but looser.

### Sensitivity Analysis

`calibration.py` also provides functions for analyzing how ε_audit
changes as a function of:
- `β` (confidence level) — `sensitivity_over_beta()`
- `δ` — `sensitivity_over_delta()`
- `r` (number of guesses) — `sensitivity_over_r()`

---

## Other Auditors (stubs)

| Auditor | File | Reference | Status |
|---|---|---|---|
| Worst-case initialization | `worst_case_init.py` | Annamalai & De Cristofaro (2024) | 🔲 stub |
| Poisoning-based | `poisoning.py` | Jagielski, Ullman & Wagner (2020) | 🔲 stub |
| DP-Sniper | `dpsniper.py` | Bichsel, Steffen & Vechev (2021) | 🔲 stub |
