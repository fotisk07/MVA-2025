# Interactions

Project code for interacting particle systems and systemic risk experiments.

## Contents

- `src/mfnn_control/`: core implementation for systemic risk and ABM experiments.
- `scripts/`: training, plotting, and ABM experiment scripts.
- `notebooks/`: exploratory and reproducibility notebooks.
- `results/`: generated checkpoints, figures, and experiment outputs.
- `report/`: report sources and compiled PDF.
- `tests/`: basic tests for the baseline implementation.

## Setup

Install the dependencies from `requirements.txt` in a virtual environment:

```bash
pip install -r requirements.txt
```

Example commands:

```bash
python scripts/train_systemic_risk_baseline.py
python scripts/run_abm_experiments.py
```
