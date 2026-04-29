#%%
from pathlib import Path

from plots.plots import plot_optimal_day_ahead_offers, plot_in_sample_profit_distribution, plot_optimal_day_ahead_offers_with_avg_imbalance
from models.StepOne import DayAheadOnePriceBuilder, DayAheadTwoPriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem
# %%
builder = DayAheadOnePriceBuilder()
builder.build_objective_coefficients()

# %% Solve optimization problem
problem = LP_OptimizationProblem(builder)
problem.run()
results = problem.get_results()
print([results["variables"][f"p_DA_{i}"] for i in range(1, builder.num_hours + 1)])

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
problem2_results = problem2.get_results()
print([problem2_results["variables"][f"p_DA_{i}"] for i in range(1, builder2.num_hours + 1)])
avg_imbalance = [
	sum(float(s.imbalance[hour]) for s in builder2.scenario_list) / len(builder2.scenario_list)
	for hour in range(builder2.num_hours)
]
print("Average imbalance by hour:")
print(avg_imbalance)
# %%
fig2 = plot_optimal_day_ahead_offers(problem2, builder2, save_path=output_dir / "optimal_day_ahead_offers_two_price.html")
fig2.show(renderer="browser")
fig3 = plot_optimal_day_ahead_offers_with_avg_imbalance(problem2, builder2, save_path=output_dir / "optimal_day_ahead_offers_with_avg_imbalance.html")
fig3.show(renderer="browser")

problem2.model.write(str(output_dir / "step1_2.lp"))
# %%
