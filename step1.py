#%%
from pathlib import Path

from plots.plots import plot_optimal_day_ahead_offers, plot_in_sample_profit_distribution, plot_one_price_vs_two_price_offers, plot_profit_distribution_comparison
from models.StepOne import DayAheadOnePriceBuilder, DayAheadTwoPriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem
# %%
builder = DayAheadOnePriceBuilder()
builder.build_objective_coefficients()

# %% Solve optimization problem
problem = LP_OptimizationProblem(builder)
problem.run()
# %%

# %%
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
problem.model.write(str(output_dir / "step1_1.lp"))
fig = plot_optimal_day_ahead_offers(problem, builder, save_path=output_dir / "optimal_day_ahead_offers.pdf")
fig = plot_in_sample_profit_distribution(problem, builder, save_path=output_dir / "in_sample_profit_distribution.pdf")


# %%
builder2 = DayAheadTwoPriceBuilder()
builder2.build_objective_coefficients()
# %%

problem2 = LP_OptimizationProblem(builder2)
problem2.run()
# %%
fig2 = plot_one_price_vs_two_price_offers(problem, problem2, builder, save_path=output_dir / "one_price_vs_two_price_offers.pdf")
fig2 = plot_profit_distribution_comparison(problem, builder, problem2, builder2, save_path=output_dir / "profit_distribution_comparison.pdf")
problem2.model.write(str(output_dir / "step1_2.lp"))
# %%
