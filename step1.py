#%%
from data.data import scenarios
from models.StepOne import DayAheadOnePriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from pathlib import Path

# %%
all_scenarios = list(scenarios.values())
builder = DayAheadOnePriceBuilder(scenario_list=all_scenarios[:200], model_name="Day-Ahead One-Price Model")
builder.build_objective_coefficients()

# %% Solve optimization problem
problem = LP_OptimizationProblem(builder)
problem.run()
problem.display_results()
results = problem.get_results()



# %%
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
problem.model.write(str(output_dir / "step1_1.lp"))

