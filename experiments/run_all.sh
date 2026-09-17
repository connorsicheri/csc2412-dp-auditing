#!/usr/bin/env bash
# =================================================================
# Overnight Experiment Runner
# =================================================================
# Runs all 8 experiments sequentially: train → audit → evaluate.
#
# Usage:
#   ./experiments/run_all.sh           # Run all experiments
#   ./experiments/run_all.sh 4         # Resume from experiment 4
#   ./experiments/run_all.sh 5 5       # Run only experiment 5
#
# Each experiment outputs to results/experiments/<exp_name>/
# Logs are saved to results/experiments/<exp_name>.log
# A summary CSV is written at the end to results/experiments/summary.csv
# =================================================================
set -uo pipefail

# ---- Configuration ----
PYTHON="conda run -n base python"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

EXPERIMENTS=(
    "cifar10_cnn_e20_s07"     # E1: Deep CNN (2.5M)
    "cifar10_wrn_e10_s07"     # E2: WRN 10 epochs
    "cifar10_wrn_e20_s07"     # E3: WRN 20 epochs
    "cifar10_wrn_e40_s07"     # E4: WRN 40 epochs
    "cifar10_vit_e20_s07"     # E5: ViT-Small (3.2M)
    "cifar10_wrn_e20_s05"     # E6: WRN low noise
    "cifar10_wrn_e20_s10"     # E7: WRN high noise
)

START_FROM="${1:-1}"
END_AT="${2:-${#EXPERIMENTS[@]}}"

echo "=============================================="
echo "  DP Auditing — Overnight Experiment Suite"
echo "=============================================="
echo "  Experiments: $START_FROM to $END_AT of ${#EXPERIMENTS[@]}"
echo "  Start time:  $(date)"
echo "  Project dir: $PROJECT_DIR"
echo "=============================================="
echo ""

RESULTS_DIR="results/experiments"
mkdir -p "$RESULTS_DIR"

PASSED=0
FAILED=0
FAILED_LIST=""

for i in "${!EXPERIMENTS[@]}"; do
    exp_num=$((i + 1))

    if [ "$exp_num" -lt "$START_FROM" ] || [ "$exp_num" -gt "$END_AT" ]; then
        continue
    fi

    exp="${EXPERIMENTS[$i]}"
    config="configs/experiments/${exp}.yaml"
    output="${RESULTS_DIR}/${exp}"
    log="${RESULTS_DIR}/${exp}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${exp_num}/${#EXPERIMENTS[@]}] ${exp}"
    echo "  Config:  ${config}"
    echo "  Output:  ${output}"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$output"

    # ---- Train ----
    echo "[$(date '+%H:%M:%S')] Training..."
    if $PYTHON experiments/run_training.py \
        --config "$config" \
        --output-dir "$output" \
        2>&1 | tee "$log"; then

        # ---- Audit ----
        echo "[$(date '+%H:%M:%S')] Auditing..."
        $PYTHON experiments/run_audit.py \
            --config "$config" \
            --output-dir "$output" \
            2>&1 | tee -a "$log"

        # ---- Evaluate ----
        echo "[$(date '+%H:%M:%S')] Evaluating..."
        $PYTHON experiments/run_evaluation.py \
            --config "$config" \
            --output-dir "$output" \
            2>&1 | tee -a "$log"

        echo "[$(date '+%H:%M:%S')] ✓ Done: ${exp}"
        PASSED=$((PASSED + 1))
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAILED: ${exp} (training crashed)"
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}  - E${exp_num}: ${exp}\n"
    fi

    echo ""
done

# ---- Generate summary CSV ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating summary..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Generate cross-experiment comparison plots
$PYTHON experiments/plot_summary.py --results-dir "$RESULTS_DIR"

# Generate summary CSV
$PYTHON - <<'PYEOF'
import json, csv, os
from pathlib import Path

experiments = [
    "cifar10_cnn_e20_s07",
    "cifar10_wrn_e10_s07",
    "cifar10_wrn_e20_s07",
    "cifar10_wrn_e40_s07",
    "cifar10_vit_e20_s07",
    "cifar10_wrn_e20_s05",
    "cifar10_wrn_e20_s10",
]

rows = []
for exp in experiments:
    d = Path(f"results/experiments/{exp}")
    meta_path = d / "training_meta.json"
    audit_path = d / "audit_one_run.json"
    tight_path = d / "tightness_metrics.json"

    if not meta_path.exists():
        continue

    with open(meta_path) as f:
        meta = json.load(f)
    cfg = meta["config"]

    row = {
        "experiment": exp,
        "dataset": cfg["dataset"]["name"],
        "model": cfg["model"]["arch"],
        "epochs": cfg["training"]["epochs"],
        "noise_mult": cfg["training"]["noise_multiplier"],
        "num_canaries": cfg["auditing"]["one_run"]["num_canaries"],
        "test_acc": f"{meta.get('test_accuracy', 0):.4f}",
        "eps_theoretical": f"{meta.get('final_epsilon', 0):.4f}" if meta.get("final_epsilon") else "N/A",
        "eps_audit": "N/A",
        "v": "N/A",
        "r": "N/A",
        "p_hat": "N/A",
        "gap_zcdp": "N/A",
        "ratio_zcdp": "N/A",
    }

    if audit_path.exists():
        with open(audit_path) as f:
            audit = json.load(f)
        row["eps_audit"] = f"{audit.get('eps_audit', 0):.4f}" if audit.get("eps_audit") else "0.0000"
        row["v"] = str(audit.get("v", ""))
        row["r"] = str(audit.get("r", ""))
        row["p_hat"] = f"{audit.get('p_hat', 0):.4f}" if audit.get("p_hat") else "N/A"

    if tight_path.exists():
        with open(tight_path) as f:
            tight = json.load(f)
        for entry in tight:
            if entry["accountant"] == "zcdp":
                row["gap_zcdp"] = f"{entry['gap']:.4f}"
                row["ratio_zcdp"] = f"{entry['ratio']:.4f}"

    rows.append(row)

if rows:
    summary_path = "results/experiments/summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved to {summary_path}")

    # Pretty-print table
    print()
    print(f"{'Experiment':<28} {'Model':<6} {'Ep':>3} {'σ':>4} {'Test':>6} {'ε_theo':>8} {'ε_audit':>8} {'v/r':>8} {'Gap':>8}")
    print("─" * 95)
    for r in rows:
        vr = f"{r['v']}/{r['r']}" if r['v'] != 'N/A' else 'N/A'
        print(f"{r['experiment']:<28} {r['model']:<6} {r['epochs']:>3} {r['noise_mult']:>4} {r['test_acc']:>6} {r['eps_theoretical']:>8} {r['eps_audit']:>8} {vr:>8} {r['gap_zcdp']:>8}")
else:
    print("No experiment results found yet.")
PYEOF

echo ""
echo "=============================================="
echo "  Experiment Suite Complete"
echo "  Finished: $(date)"
echo "  Passed:   ${PASSED}"
echo "  Failed:   ${FAILED}"
if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  Failed experiments:"
    echo -e "$FAILED_LIST"
fi
echo "=============================================="
