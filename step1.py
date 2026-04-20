#%%
from data.data import scenarios
from models.StepOne import DayAheadOnePriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from pathlib import Path

#%%
scenarios
# %%
builder = DayAheadOnePriceBuilder()
builder.build_objective_coefficients()

# %% Solve optimization problem
problem = LP_OptimizationProblem(builder)

# %%
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
problem.model.write(str(output_dir / "step1_1.lp"))

