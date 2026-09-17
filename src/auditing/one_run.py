"""One-run canary auditing (Steinke, Nasr, Jagielski 2023).

Implements the *mislabeled-example canary* strategy from Section 4 of:

    Steinke, Nasr, Jagielski. "Privacy Auditing with One (1) Training Run."
    NeurIPS 2023.  arXiv:2305.08846.

The auditor:
    1. Samples m examples from the real training set.
    2. Replaces each true label y with a uniformly random *incorrect* label
       ỹ ≠ y (the "canary" is the mislabeled copy).
    3. Flips m independent fair coins b_i ∈ {0, 1}.
       • b_i = 1 → the canary is INCLUDED in the training set.
       • b_i = 0 → the canary is EXCLUDED.
    4. Trains the model once on (base data ∪ included canaries).
    5. Scores every canary with the trained model:
           score_i = −loss(model, (x_i, ỹ_i))
       Intuition: if the canary was included, the model tried to fit the
       wrong label ỹ, so loss is low → score is high.
    6. Guesses b̂_i = IN if score > threshold, OUT otherwise (or abstains).
    7. Counts v = #correct guesses out of r = #non-abstained, then inverts
       the binomial tail to obtain a calibrated ε lower bound.

Why mislabeled real examples (not random noise)?
    Random Gaussian noise lies far from the data manifold.  Models cannot
    learn to fit noise images even without DP, so IN/OUT scores are
    indistinguishable → v ≈ r/2 → ε_audit = 0.  Mislabeled real examples
    sit on the manifold but conflict with the true labels of nearby points,
    giving the model a memorization signal that DP should suppress.

References
----------
Steinke, Nasr, Jagielski (2023). Privacy Auditing with One (1) Training Run.
    NeurIPS 2023. arXiv:2305.08846.

Keinan, Shenfeld, Ligett (2025). How Well Can DP Be Audited in One Run?
    NeurIPS 2025. arXiv:2503.07199.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .calibration import CalibrationResult, compute_eps_lower_bound

logger = logging.getLogger(__name__)


class _CanaryDataset(Dataset):
    """Dataset of mislabeled copies of real training examples.

    Each item is (x_i, ỹ_i) where x_i is the original image and ỹ_i ≠ y_i
    is a uniformly random incorrect label.  Returns ``(Tensor, int)`` to
    match torchvision dataset conventions (avoids collation issues when
    concatenated with MNIST / CIFAR datasets that return ``int`` labels).

    Attributes
    ----------
    images : Tensor   – canary images (copied from real data).
    labels : Tensor   – mislabeled targets (ỹ).
    original_labels : Tensor – the true labels y (kept for diagnostics).
    source_indices : ndarray – indices into the base dataset that were sampled.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        original_labels: torch.Tensor,
        source_indices: np.ndarray,
    ):
        self.images = images
        self.labels = labels
        self.original_labels = original_labels
        self.source_indices = source_indices

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], int(self.labels[idx])


@dataclass
class CanaryResult:
    """Result of one-run canary auditing."""

    v: int  # correct guesses
    r: int  # non-abstained guesses
    m: int  # total canaries
    scores: np.ndarray  # per-canary scores
    included: np.ndarray  # boolean mask: was canary included?
    guesses: np.ndarray  # +1 (IN), -1 (OUT), 0 (abstain)
    calibration: CalibrationResult | None = None


class OneRunCanaryAuditor:
    """One-run canary auditor for DP-SGD (Steinke et al. 2023).

    Canaries are **mislabeled copies of real training examples** (Section 4
    of the paper).  Each true label y is replaced with a uniform random
    incorrect label ỹ ≠ y, and an independent coin flip decides whether
    the canary is included in the training set.

    Parameters
    ----------
    num_canaries : int
        Number of canary examples *m* to create.
    abstention_threshold : float
        Minimum |score| to make a guess.  Scores below this → abstain.
        Set to 0 for no abstention (median-split decision boundary).
    confidence_beta : float
        Confidence parameter β for the calibrated ε lower bound.
    delta : float
        The δ in (ε, δ)-DP.
    seed : int
        Random seed for reproducibility (canary sampling, label flipping,
        coin-flip inclusion).
    """

    def __init__(
        self,
        num_canaries: int = 500,
        abstention_threshold: float = 0.0,
        confidence_beta: float = 0.05,
        delta: float = 1e-5,
        seed: int = 42,
    ):
        self.num_canaries = num_canaries
        self.abstention_threshold = abstention_threshold
        self.confidence_beta = confidence_beta
        self.delta = delta
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Canary generation  (Steinke et al. 2023 §4 — mislabeled examples)
    # ------------------------------------------------------------------

    def generate_canaries(
        self,
        base_dataset: Dataset,
        num_classes: int = 10,
        mislabel: bool = True,
    ) -> tuple[np.ndarray, _CanaryDataset]:
        """Sample real examples to create canaries.

        Two modes:
        - ``mislabel=True`` (default): each true label y is replaced with a
          uniform random incorrect label ỹ ≠ y.  Best with m << n.
        - ``mislabel=False``: canaries keep their correct labels.  The
          membership signal comes from the model overfitting to training
          data (membership inference).  Best with m = n.

        If ``self.num_canaries == 0``, m is set to len(base_dataset) (m = n).

        Parameters
        ----------
        base_dataset : Dataset
            The base training dataset.
        num_classes : int
            Number of classes K.
        mislabel : bool
            If True, replace labels with random incorrect labels.
            If False, keep correct labels (membership inference mode).

        Returns
        -------
        included : np.ndarray of bool, shape ``(m,)``
            Random inclusion mask (each canary included w.p. ½).
        canary_dataset : _CanaryDataset
            The canary examples.
        """
        n = len(base_dataset)
        m = n if self.num_canaries == 0 else self.num_canaries
        self.num_canaries = m  # store actual value for calibration

        # 1. Sample m indices (all indices if m == n)
        if m == n:
            source_indices = np.arange(n)
        else:
            source_indices = self.rng.choice(n, size=m, replace=(m > n))

        images = []
        original_labels = []
        for idx in source_indices:
            x, y = base_dataset[int(idx)]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            images.append(x)
            original_labels.append(int(y))

        images = torch.stack(images)
        original_labels_t = torch.tensor(original_labels, dtype=torch.long)

        # 2. Labels: mislabel or keep correct
        if mislabel:
            canary_labels = torch.empty_like(original_labels_t)
            for i, y in enumerate(original_labels):
                candidates = [c for c in range(num_classes) if c != y]
                canary_labels[i] = int(self.rng.choice(candidates))
            label_desc = "mislabeled"
        else:
            canary_labels = original_labels_t.clone()
            label_desc = "correctly-labeled"

        # 3. Random coin-flip inclusion: b_i ~ Bernoulli(1/2)
        included = self.rng.random(m) < 0.5

        canary_dataset = _CanaryDataset(images, canary_labels, original_labels_t, source_indices)

        logger.info(
            f"Generated {m} {label_desc} canaries from {n} training examples "
            f"({int(included.sum())} included, m={'n' if m == n else m})"
        )

        return included, canary_dataset

    def build_training_set(
        self,
        base_dataset: Dataset,
        canary_dataset: Dataset,
        included: np.ndarray,
        include_base: bool = True,
    ) -> Dataset:
        """Build the training dataset.

        Two modes:
        - ``include_base=True`` (default): base data ∪ included canaries.
          Use when m << n (mislabeled canary mode).
        - ``include_base=False``: ONLY included canaries, no base data.
          Use when m = n (membership inference mode).  This matches the
          paper's §6.2 black-box setting where all data are canaries.

        Parameters
        ----------
        base_dataset : Dataset
            Original training data (correctly labeled).
        canary_dataset : _CanaryDataset
            All m canary examples.
        included : np.ndarray of bool
            Coin-flip mask.  ``included[i] == True`` means canary i is IN.
        include_base : bool
            If True, prepend the full base dataset.  If False, train only
            on included canaries (m = n setting).

        Returns
        -------
        Dataset for training.
        """
        from torch.utils.data import ConcatDataset

        included_indices = np.where(included)[0]
        if len(included_indices) == 0:
            logger.warning("No canaries included — returning base dataset")
            return base_dataset

        included_canaries = Subset(canary_dataset, included_indices.tolist())

        if include_base:
            return ConcatDataset([base_dataset, included_canaries])
        else:
            logger.info(
                f"Canary-only mode: training on {len(included_indices)} "
                f"included canaries (no base data)"
            )
            return included_canaries

    def score_canaries(
        self,
        model: nn.Module,
        canary_dataset: Dataset,
        device: str = "cpu",
        init_model: nn.Module | None = None,
    ) -> np.ndarray:
        """Score each canary using the trained model.

        Paper's black-box scoring (Algorithm 3, line 9):
            score_i = loss(w0, x_i, ỹ_i) − loss(w_final, x_i, ỹ_i)

        If init_model is provided, we compute the full paper score.
        Otherwise, we fall back to score_i = −loss(w_final, x_i, ỹ_i).

        Intuition: included canaries have low final loss (model memorized
        the wrong label) → large positive score.

        Parameters
        ----------
        model : nn.Module
            Trained model (after DP-SGD).
        canary_dataset : _CanaryDataset
            All m canaries (mislabeled copies).
        device : str
            Torch device.
        init_model : nn.Module, optional
            Model at initialization (w0).  If provided, computes the
            paper's exact score: loss(w0, x) − loss(w_final, x).

        Returns
        -------
        scores : np.ndarray, shape ``(num_canaries,)``
            Higher → more likely included.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="none")
        loader = DataLoader(canary_dataset, batch_size=256, shuffle=False)

        final_losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                losses = criterion(logits, y)
                final_losses.append(losses.cpu().numpy())
        final_losses = np.concatenate(final_losses)

        if init_model is not None:
            # Paper's score: loss(w0) - loss(w_final)
            init_model.eval()
            init_losses = []
            with torch.no_grad():
                for x, y in DataLoader(canary_dataset, batch_size=256, shuffle=False):
                    x, y = x.to(device), y.to(device)
                    logits = init_model(x)
                    losses = criterion(logits, y)
                    init_losses.append(losses.cpu().numpy())
            init_losses = np.concatenate(init_losses)
            return init_losses - final_losses
        else:
            # Fallback: -loss(w_final)
            return -final_losses

    def make_guesses(
        self,
        scores: np.ndarray,
        k_plus: int | None = None,
        k_minus: int | None = None,
    ) -> np.ndarray:
        """Make inclusion guesses based on scores.

        Paper strategy (Algorithm 1): guess IN for the top k+ scores,
        OUT for the bottom k- scores, abstain on the rest.

        If k_plus and k_minus are not given, falls back to median split
        (equivalent to k+ = k- = m/2, no abstentions).

        Parameters
        ----------
        scores : np.ndarray
            Per-canary scores.
        k_plus : int, optional
            Number of top scores to guess IN.
        k_minus : int, optional
            Number of bottom scores to guess OUT.

        Returns
        -------
        guesses : np.ndarray of {+1, -1, 0}
        """
        m = len(scores)
        guesses = np.zeros(m, dtype=int)

        if k_plus is not None and k_minus is not None:
            # Paper's strategy: top k+ → IN, bottom k- → OUT
            sorted_idx = np.argsort(scores)
            top_idx = sorted_idx[-k_plus:] if k_plus > 0 else []
            bot_idx = sorted_idx[:k_minus] if k_minus > 0 else []
            guesses[top_idx] = 1
            guesses[bot_idx] = -1
        elif self.abstention_threshold == 0:
            # No abstention: median split
            median = np.median(scores)
            guesses[scores > median] = 1
            guesses[scores <= median] = -1
        else:
            guesses[scores > self.abstention_threshold] = 1
            guesses[scores < -self.abstention_threshold] = -1

        return guesses

    def evaluate_guesses(self, guesses: np.ndarray, included: np.ndarray) -> tuple[int, int]:
        """Count correct guesses and total non-abstained guesses.

        Parameters
        ----------
        guesses : np.ndarray of {+1, -1, 0}
        included : np.ndarray of bool

        Returns
        -------
        v : int — number of correct guesses
        r : int — number of non-abstained guesses
        """
        # Non-abstained indices
        active = guesses != 0
        r = int(active.sum())

        # Ground truth: +1 if included, -1 if excluded
        truth = np.where(included, 1, -1)

        # Correct if guess matches truth
        v = int(np.sum((guesses == truth) & active))

        return v, r

    def audit(
        self,
        model: nn.Module,
        canary_dataset: Dataset,
        included: np.ndarray,
        device: str = "cpu",
        init_model: nn.Module | None = None,
    ) -> CanaryResult:
        """Run the full one-run canary audit pipeline (Steinke et al. 2023).

        Steps:
            1. Score every canary with the trained (and optionally init) model.
            2. Try multiple k+/k- guessing strategies (paper evaluates several
               and reports the best, per §6).
            3. Invert the binomial tail to get ε_audit at confidence 1−β.

        Parameters
        ----------
        model : nn.Module
            Trained model (after DP-SGD on base data ∪ included canaries).
        canary_dataset : _CanaryDataset
            All m mislabeled canary examples.
        included : np.ndarray of bool
            Ground-truth coin-flip inclusion mask.
        device : str
            Torch device.
        init_model : nn.Module, optional
            Model at initialization for paper's score: loss(w0) - loss(w_final).

        Returns
        -------
        CanaryResult
            Contains v, r, scores, guesses, and calibrated ε lower bound.
        """
        m = self.num_canaries

        # Step 1: Score canaries
        scores = self.score_canaries(model, canary_dataset, device, init_model=init_model)

        # Step 2: Try multiple k+/k- strategies and pick the best ε_audit
        # (Paper §6: "we evaluate different values of k+ and k- and only
        # report the highest auditing results")
        best_eps = -1.0
        best_v, best_r, best_guesses = 0, 0, np.zeros(m, dtype=int)
        best_cal = None

        # Candidate k values: symmetric k+ = k- = k
        k_candidates = sorted(
            set(
                [
                    m,  # no abstention (median split)
                    int(0.9 * m),
                    int(0.75 * m),
                    int(0.5 * m),
                    int(0.25 * m),
                    int(0.1 * m),
                    int(0.05 * m),
                    max(10, int(0.01 * m)),
                ]
            )
        )

        for k in k_candidates:
            kp = k // 2
            km = k - kp
            if kp == 0 or km == 0:
                continue

            guesses = self.make_guesses(scores, k_plus=kp, k_minus=km)
            v, r = self.evaluate_guesses(guesses, included)
            if r == 0 or v <= r // 2:
                continue

            cal = compute_eps_lower_bound(
                v=v,
                r=r,
                beta=self.confidence_beta,
                delta=self.delta,
                m=m,
            )
            if cal.eps_lower_bound > best_eps:
                best_eps = cal.eps_lower_bound
                best_v, best_r = v, r
                best_guesses = guesses
                best_cal = cal

        # Also try the simple median split (k+ = k- = m/2)
        guesses = self.make_guesses(scores)
        v, r = self.evaluate_guesses(guesses, included)
        if r > 0 and v > r // 2:
            cal = compute_eps_lower_bound(
                v=v,
                r=r,
                beta=self.confidence_beta,
                delta=self.delta,
                m=m,
            )
            if cal.eps_lower_bound > best_eps:
                best_eps = cal.eps_lower_bound
                best_v, best_r = v, r
                best_guesses = guesses
                best_cal = cal

        if best_cal is None:
            best_cal = compute_eps_lower_bound(
                v=0,
                r=m,
                beta=self.confidence_beta,
                delta=self.delta,
                m=m,
            )
            best_v, best_r = v, r
            best_guesses = guesses

        logger.info(
            f"One-run audit: v={best_v} correct out of r={best_r} guesses "
            f"(m={m} canaries, p_hat={best_v/max(best_r,1):.3f})"
        )
        logger.info(
            f"ε_audit = {best_cal.eps_lower_bound:.4f} "
            f"(β={self.confidence_beta}, δ={self.delta})"
        )

        return CanaryResult(
            v=best_v,
            r=best_r,
            m=m,
            scores=scores,
            included=included,
            guesses=best_guesses,
            calibration=best_cal,
        )

    def quick_audit(
        self,
        model: nn.Module,
        canary_dataset: Dataset,
        included: np.ndarray,
        device: str = "cpu",
        init_model: nn.Module | None = None,
    ) -> float:
        """Fast per-epoch audit: score canaries → return ε_audit.

        Uses a single median-split strategy (no k+/k- search) for speed.
        Suitable for tracking ε_audit over epochs during training.

        Returns
        -------
        eps_audit : float
            Calibrated ε lower bound (0.0 if no signal).
        """
        m = self.num_canaries
        scores = self.score_canaries(model, canary_dataset, device, init_model=init_model)
        guesses = self.make_guesses(scores)
        v, r = self.evaluate_guesses(guesses, included)
        if r == 0 or v <= r // 2:
            return 0.0
        cal = compute_eps_lower_bound(
            v=v,
            r=r,
            beta=self.confidence_beta,
            delta=self.delta,
            m=m,
        )
        return cal.eps_lower_bound
