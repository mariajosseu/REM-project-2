# REM Project 2 — Renewable Market Models
Optimal Market Participation Strategies for Wind Power and Flexible Loads in Electricity Markets
## What this repo contains

- Main scripts: `step1.py`, `step2.py`, `utils.py`
- Input datasets and profiles: `data/`
- Optimization models and classes: `models/` (e.g., `OptimizationClasses.py`, `StepOne.py`, `StepTwo.py`)
- Generated results and LP files: `outputs/`
- Plotting helpers: `plots/`

## Quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: Gurobi requires a valid license.

## Run

Run any step directly:

```bash
python step1.py
python step2.py
```

`utils.py` contains helper functions used by the main steps.

## Outputs

- LP model exports and model files are saved in `outputs/`.
- Numerical results (CSV) are stored in `outputs/`.
- Figures are saved to `outputs/images/` when plotting is enabled.


## Class logic (short)

The code follows a builder -> solver -> analysis flow:

- Step builders (`StepOne`, `StepTwo`) create the mathematical model inputs for each assignment step.
- `LP_InputData` (in `OptimizationClasses.py`) stores the model definition: variables, objective coefficients, constraint coefficients, right-hand sides, and senses.
- `LP_OptimizationProblem` builds and solves the Gurobi model from `LP_InputData`.
- `plots/Plots.py` is used for post-processing and visualization.

Typical run flow in any `stepX.py` script:

1. Build the step-specific input data with a Builder class.
2. Pass that data into `LP_OptimizationProblem`.
3. Solve with Gurobi.
4. Print summaries and optionally export LP/CSV/plots.