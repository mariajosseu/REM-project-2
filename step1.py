#%%
from data.data import scenarios
from models.StepOne import DayAheadOnePriceBuilder

#%%
scenarios
# %%
builder = DayAheadOnePriceBuilder()
builder.build_objective_coefficients()
# %%