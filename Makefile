.PHONY: install test lint train audit eval all clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ experiments/
	black --check src/ tests/ experiments/

format:
	black src/ tests/ experiments/

# --- Experiments ---

train-mnist-logreg:
	python experiments/run_training.py --config configs/mnist_logreg.yaml

audit-mnist-logreg:
	python experiments/run_audit.py --config configs/mnist_logreg.yaml --auditor one_run

eval-mnist-logreg:
	python experiments/run_evaluation.py --config configs/mnist_logreg.yaml

# Full pipeline for one config
run-mnist-logreg: train-mnist-logreg audit-mnist-logreg eval-mnist-logreg

train-mnist-mlp:
	python experiments/run_training.py --config configs/mnist_mlp.yaml --output-dir results/mnist_mlp

audit-mnist-mlp:
	python experiments/run_audit.py --config configs/mnist_mlp.yaml --output-dir results/mnist_mlp

eval-mnist-mlp:
	python experiments/run_evaluation.py --config configs/mnist_mlp.yaml --output-dir results/mnist_mlp

run-mnist-mlp: train-mnist-mlp audit-mnist-mlp eval-mnist-mlp

train-cifar10-cnn:
	python experiments/run_training.py --config configs/cifar10_cnn.yaml --output-dir results/cifar10_cnn

audit-cifar10-cnn:
	python experiments/run_audit.py --config configs/cifar10_cnn.yaml --output-dir results/cifar10_cnn

eval-cifar10-cnn:
	python experiments/run_evaluation.py --config configs/cifar10_cnn.yaml --output-dir results/cifar10_cnn

run-cifar10-cnn: train-cifar10-cnn audit-cifar10-cnn eval-cifar10-cnn

# CIFAR-10 + WRN-16 (paper reproduction)
train-cifar10-wrn:
	python experiments/run_training.py --config configs/cifar10_wrn.yaml --output-dir results/cifar10_wrn

audit-cifar10-wrn:
	python experiments/run_audit.py --config configs/cifar10_wrn.yaml --output-dir results/cifar10_wrn

eval-cifar10-wrn:
	python experiments/run_evaluation.py --config configs/cifar10_wrn.yaml --output-dir results/cifar10_wrn

run-cifar10-wrn: train-cifar10-wrn audit-cifar10-wrn eval-cifar10-wrn

# --- Overnight experiment suite ---
run-all-experiments:
	./experiments/run_all.sh

# Resume from experiment N (e.g., make resume-from EXP=4)
resume-from:
	./experiments/run_all.sh $(EXP)

# Run a single experiment (e.g., make run-single EXP=5)
run-single:
	./experiments/run_all.sh $(EXP) $(EXP)

# Generate summary table from completed experiments
experiment-summary:
	@python - <<< 'exec(open("experiments/run_all.sh").read().split("PYEOF")[1].split("PYEOF")[0])' 2>/dev/null || echo "Run experiments first."

clean:
	rm -rf results/ data/ __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
