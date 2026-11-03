#!/bin/bash

set -e

PROJECT_ROOT="brainsentinel"

echo "Creating project structure for $PROJECT_ROOT..."

# Create root directories
mkdir -p "$PROJECT_ROOT/configs"
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/notebooks"
mkdir -p "$PROJECT_ROOT/src/brainsentinel/data"
mkdir -p "$PROJECT_ROOT/src/brainsentinel/models"
mkdir -p "$PROJECT_ROOT/src/brainsentinel/evaluation"
mkdir -p "$PROJECT_ROOT/src/brainsentinel/inference"
mkdir -p "$PROJECT_ROOT/src/brainsentinel/agents"
mkdir -p "$PROJECT_ROOT/src/brainsentinel/api"
mkdir -p "$PROJECT_ROOT/tests"
mkdir -p "$PROJECT_ROOT/scripts"
mkdir -p "$PROJECT_ROOT/artifacts/checkpoints"
mkdir -p "$PROJECT_ROOT/artifacts/heatmaps"
mkdir -p "$PROJECT_ROOT/artifacts/figures"
mkdir -p "$PROJECT_ROOT/artifacts/reports"
mkdir -p "$PROJECT_ROOT/results"

touch "$PROJECT_ROOT/requirements.txt"
touch "$PROJECT_ROOT/pyproject.toml"
touch "$PROJECT_ROOT/README.md"
touch "$PROJECT_ROOT/.gitignore"

# Create configs files
touch "$PROJECT_ROOT/configs/data.yaml"
touch "$PROJECT_ROOT/configs/patchcore.yaml"
touch "$PROJECT_ROOT/configs/padim.yaml"
touch "$PROJECT_ROOT/configs/autoencoder.yaml"

# Create data README
cat << 'EOF' > "$PROJECT_ROOT/data/README.md"
# Data Directory
Source, license, counts, and checksums for datasets used in BrainSentinel.
EOF

# Create notebooks (minimal valid JSON notebooks)
echo '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}' > "$PROJECT_ROOT/notebooks/01_dataset_exploration.ipynb"
echo '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}' > "$PROJECT_ROOT/notebooks/02_error_analysis.ipynb"

# Create package init files and source modules
touch "$PROJECT_ROOT/src/brainsentinel/__init__.py"
touch "$PROJECT_ROOT/src/brainsentinel/data/__init__.py"
touch "$PROJECT_ROOT/src/brainsentinel/data/dataset.py"
touch "$PROJECT_ROOT/src/brainsentinel/data/preprocessing.py"
touch "$PROJECT_ROOT/src/brainsentinel/data/checks.py"

touch "$PROJECT_ROOT/src/brainsentinel/models/__init__.py"
touch "$PROJECT_ROOT/src/brainsentinel/models/patchcore_model.py"
touch "$PROJECT_ROOT/src/brainsentinel/models/padim_model.py"
touch "$PROJECT_ROOT/src/brainsentinel/models/autoencoder.py"

touch "$PROJECT_ROOT/src/brainsentinel/evaluation/__init__.py"
touch "$PROJECT_ROOT/src/brainsentinel/evaluation/metrics.py"
touch "$PROJECT_ROOT/src/brainsentinel/evaluation/thresholding.py"
touch "$PROJECT_ROOT/src/brainsentinel/evaluation/plots.py"

touch "$PROJECT_ROOT/src/brainsentinel/inference/__init__.py"
touch "$PROJECT_ROOT/src/brainsentinel/inference/predict.py"

touch "$PROJECT_ROOT/src/brainsentinel/agents/__init__.py"
touch "$PROJECT_ROOT/src/brainsentinel/agents/graph.py"
touch "$PROJECT_ROOT/src/brainsentinel/agents/quality_agent.py"
touch "$PROJECT_ROOT/src/brainsentinel/agents/evidence_agent.py"
touch "$PROJECT_ROOT/src/brainsentinel/agents/report_agent.py"

touch "$PROJECT_ROOT/src/brainsentinel/api/__init__.py"
touch "$PROJECT_ROOT/src/brainsentinel/api/main.py"

# Create tests
touch "$PROJECT_ROOT/tests/__init__.py"
touch "$PROJECT_ROOT/tests/test_dataset.py"
touch "$PROJECT_ROOT/tests/test_preprocessing.py"
touch "$PROJECT_ROOT/tests/test_metrics.py"
touch "$PROJECT_ROOT/tests/test_inference.py"

# Create scripts
touch "$PROJECT_ROOT/scripts/inspect_dataset.py"
touch "$PROJECT_ROOT/scripts/inspect_dataset.py"
touch "$PROJECT_ROOT/scripts/train.py"
touch "$PROJECT_ROOT/scripts/evaluate.py"
touch "$PROJECT_ROOT/scripts/predict_one.py"

# Create .gitkeep placeholders for empty directories so git tracks them if needed
touch "$PROJECT_ROOT/data/raw/.gitkeep"
touch "$PROJECT_ROOT/data/processed/.gitkeep"
touch "$PROJECT_ROOT/artifacts/checkpoints/.gitkeep"
touch "$PROJECT_ROOT/artifacts/heatmaps/.gitkeep"
touch "$PROJECT_ROOT/artifacts/figures/.gitkeep"
touch "$PROJECT_ROOT/artifacts/reports/.gitkeep"

# Create initial results files
touch "$PROJECT_ROOT/results/metrics.csv"
touch "$PROJECT_ROOT/results/experiment_log.csv"
echo "# Model Comparison" > "$PROJECT_ROOT/results/model_comparison.md"

echo "Successfully scaffolded the '$PROJECT_ROOT' project structure!"