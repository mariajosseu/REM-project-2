#%%
from data.data import scenarios
from models.StepOne import DayAheadOnePriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem

#%%
scenarios
# %%
builder = DayAheadOnePriceBuilder()
builder.build_objective_coefficients()

# %% Solve optimization problem
problem = LP_OptimizationProblem(builder)

# %%

