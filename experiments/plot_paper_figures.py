"""Generate all figures for the paper from experiment results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def load_meta(filename):
    with open(DATA / filename) as f:
        return json.load(f)


# ===== Load all data =====

# New experiments: same σ=3.0, same ε budget, different models
logreg = load_meta("eps8_logreg_25ep.json")
mlp = load_meta("eps8_mlp_25ep.json")
wrn_eps8 = load_meta("eps8_wrn_mislabel_25ep.json")

# Long WRN run: audit eventually works at high ε
wrn_long = load_meta("wrn_s1_b500_1000ep.json")

# Non-DP baseline
nondp = load_meta("wrn_nondp_baseline.json")


# ===== Figure 1: ε_audit vs ε_account across models =====
fig, ax = plt.subplots(figsize=(6.5, 4.5))

# LogReg
eps_pld_lr = logreg["epsilon_history"]["pld"]
eps_aud_lr = logreg["eps_audit_history"]
ax.plot(
    eps_pld_lr,
    eps_aud_lr,
    "o-",
    color="#2196F3",
    markersize=4,
    linewidth=1.5,
    label="LogReg (31K params)",
)

# MLP
eps_pld_mlp = mlp["epsilon_history"]["pld"]
eps_aud_mlp = mlp["eps_audit_history"]
ax.plot(
    eps_pld_mlp,
    eps_aud_mlp,
    "s-",
    color="#4CAF50",
    markersize=4,
    linewidth=1.5,
    label="MLP (395K params)",
)

# WRN at ε≈8
eps_pld_wrn = wrn_eps8["epsilon_history"]["pld"]
eps_aud_wrn = wrn_eps8["eps_audit_history"]
ax.plot(
    [eps_pld_wrn[-1]],
    [eps_aud_wrn[-1]],
    "^",
    color="#F44336",
    markersize=6,
    label="WRN-16-4 (2.75M params)",
)

# Diagonal (tightness = 1)
max_eps = max(max(eps_pld_lr), max(eps_pld_mlp))
ax.plot(
    [0, max_eps],
    [0, max_eps],
    "k--",
    alpha=0.3,
    linewidth=1,
    label="Tight ($\\varepsilon_{\\mathrm{audit}} = \\varepsilon_{\\mathrm{account}}$)",
)

ax.set_xlabel("$\\varepsilon_{\\mathrm{account}}$ (PLD upper bound)")
ax.set_ylabel("$\\varepsilon_{\\mathrm{audit}}$ (one-run lower bound)")
ax.set_title("Audit Tightness vs. Model Capacity ($\\sigma=3.0$, $m=n=1000$)")
ax.legend(loc="upper left")
ax.set_xlim(0, 9)
ax.set_ylim(-0.05, 1.0)
ax.grid(True, alpha=0.3)
fig.savefig(FIGURES / "fig1_audit_vs_account_by_model.pdf")
fig.savefig(FIGURES / "fig1_audit_vs_account_by_model.png")
plt.close(fig)
print("Saved fig1_audit_vs_account_by_model")


# ===== Figure 2: Training loss trajectories =====
fig, ax = plt.subplots(figsize=(6.5, 4))

ax.plot(
    range(1, 26),
    logreg["train_losses"],
    "o-",
    color="#2196F3",
    markersize=3,
    linewidth=1.5,
    label="LogReg (31K)",
)
ax.plot(
    range(1, 26),
    mlp["train_losses"],
    "s-",
    color="#4CAF50",
    markersize=3,
    linewidth=1.5,
    label="MLP (395K)",
)
ax.plot(
    range(1, 26),
    wrn_eps8["train_losses"][:25],
    "^-",
    color="#F44336",
    markersize=3,
    linewidth=1.5,
    label="WRN-16-4 (2.75M)",
)
ax.axhline(y=np.log(10), color="gray", linestyle=":", alpha=0.5, label="Random ($\\ln 10$)")

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss Under DP-SGD ($\\sigma=3.0$, $B=500$)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(FIGURES / "fig2_training_loss_by_model.pdf")
fig.savefig(FIGURES / "fig2_training_loss_by_model.png")
plt.close(fig)
print("Saved fig2_training_loss_by_model")


# ===== Figure 3: WRN long run — audit trajectory =====
fig, ax1 = plt.subplots(figsize=(6.5, 4.5))

eps_pld_long = wrn_long["epsilon_history"]["pld"]
eps_aud_long = wrn_long["eps_audit_history"]
epochs_long = wrn_long["eval_epochs"]

color1 = "#F44336"
color2 = "#2196F3"

ax1.set_xlabel("Epoch")
ax1.set_ylabel("$\\varepsilon_{\\mathrm{account}}$ (PLD)", color=color1)
ax1.plot(
    epochs_long,
    eps_pld_long,
    "-",
    color=color1,
    linewidth=1.5,
    label="$\\varepsilon_{\\mathrm{account}}$",
)
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
ax2.set_ylabel("$\\varepsilon_{\\mathrm{audit}}$", color=color2)
ax2.plot(
    epochs_long,
    eps_aud_long,
    "o-",
    color=color2,
    markersize=4,
    linewidth=1.5,
    label="$\\varepsilon_{\\mathrm{audit}}$",
)
ax2.tick_params(axis="y", labelcolor=color2)

# Mark ε=8 reference
ax1.axhline(y=8, color="gray", linestyle=":", alpha=0.5)
ax1.annotate("$\\varepsilon=8$", xy=(50, 8), fontsize=9, color="gray")

ax1.set_title("WRN-16-4: Audit Signal Requires Extended Training ($\\sigma=1.0$)")
fig.tight_layout()
fig.savefig(FIGURES / "fig3_wrn_long_run_trajectory.pdf")
fig.savefig(FIGURES / "fig3_wrn_long_run_trajectory.png")
plt.close(fig)
print("Saved fig3_wrn_long_run_trajectory")


# ===== Figure 4: SNR analysis =====
fig, ax = plt.subplots(figsize=(6, 4))

models = ["LogReg\n(31K)", "MLP\n(395K)", "SmallCNN\n(~545K)", "WRN-16-4\n(2.75M)"]
params = [31000, 395000, 545000, 2750000]
sigma, C, B = 3.0, 1.0, 500
snr = [B / (sigma * np.sqrt(d)) for d in params]

bars = ax.bar(models, snr, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"], alpha=0.8)

ax.set_ylabel("Per-Coordinate SNR = $B / (\\sigma \\sqrt{d})$")
ax.set_title("Signal-to-Noise Ratio by Model Size ($\\sigma=3.0$, $B=500$)")

for bar, s in zip(bars, snr):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.02,
        f"{s:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3, axis="y")
fig.savefig(FIGURES / "fig4_snr_by_model.pdf")
fig.savefig(FIGURES / "fig4_snr_by_model.png")
plt.close(fig)
print("Saved fig4_snr_by_model")


# ===== Summary table data =====
print("\n===== SUMMARY TABLE =====")
print(f"{'Model':<15} {'Params':<10} {'ε_PLD':>8} {'ε_audit':>10} {'Loss':>8} {'Tightness':>10}")
print("-" * 65)
# At epoch 25 (ε≈8)
print(
    f"{'LogReg':<15} {'31K':<10} {logreg['epsilon_history']['pld'][-1]:>8.2f} {logreg['eps_audit_history'][-1]:>10.4f} {logreg['train_losses'][-1]:>8.3f} {logreg['eps_audit_history'][-1]/logreg['epsilon_history']['pld'][-1]*100:>9.1f}%"
)
print(
    f"{'MLP [128]':<15} {'395K':<10} {mlp['epsilon_history']['pld'][-1]:>8.2f} {mlp['eps_audit_history'][-1]:>10.4f} {mlp['train_losses'][-1]:>8.3f} {mlp['eps_audit_history'][-1]/mlp['epsilon_history']['pld'][-1]*100:>9.1f}%"
)
print(
    f"{'WRN-16-4':<15} {'2.75M':<10} {wrn_eps8['epsilon_history']['pld'][-1]:>8.2f} {wrn_eps8['eps_audit_history'][-1]:>10.4f} {wrn_eps8['train_losses'][-1]:>8.3f} {'0.0':>10}%"
)
print()
# WRN long run
print(
    f"{'WRN (long)':<15} {'2.75M':<10} {wrn_long['epsilon_history']['pld'][-1]:>8.2f} {wrn_long['eps_audit_history'][-1]:>10.4f} {wrn_long['train_losses'][-1]:>8.3f} {wrn_long['eps_audit_history'][-1]/wrn_long['epsilon_history']['pld'][-1]*100:>9.1f}%"
)
# Non-DP
print(
    f"{'WRN (no-DP)':<15} {'2.75M':<10} {'∞':>8} {nondp['eps_audit_history'][-1]:>10.4f} {nondp['train_losses'][-1]:>8.3f} {'N/A':>10}"
)
