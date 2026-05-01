#%%
from data.data import scenarios
from pathlib import Path
import numpy as np

from plots.plots import plot_optimal_day_ahead_offers, plot_in_sample_profit_distribution, plot_profit_distribution_comparison, plot_one_price_vs_two_price_offers
from models.StepOne import DayAheadOnePriceBuilder, DayAheadTwoPriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from utils import evaluate_one_price_profit, evaluate_two_price_profit


all_scenarios = list(scenarios.values())
builder = DayAheadOnePriceBuilder(scenario_list=all_scenarios[:200], model_name="Day-Ahead One-Price Model")
builder.build_objective_coefficients()

# %% Solve optimization problem 1.1
problem = LP_OptimizationProblem(builder)
problem.run()
results = problem.get_results()
print([results["variables"][f"p_DA_{i}"] for i in range(1, builder.num_hours + 1)])

# %%
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
problem.model.write(str(output_dir / "step1_1.lp"))
fig = plot_optimal_day_ahead_offers(problem, builder, save_path=output_dir / "optimal_day_ahead_offers.pdf")
fig = plot_in_sample_profit_distribution(problem, builder, save_path=output_dir / "in_sample_profit_distribution.pdf")


# %% Solve optimization problem 1.2
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
fig2 = plot_one_price_vs_two_price_offers(problem, problem2, builder, save_path=output_dir / "one_price_vs_two_price_offers.pdf")
fig2 = plot_profit_distribution_comparison(problem, builder, problem2, builder2, save_path=output_dir / "profit_distribution_comparison.pdf")
problem2.model.write(str(output_dir / "step1_2.lp"))

# %% Solve optimization problem 1.3 (cross-validation)
all_scenarios = list(scenarios.values())[:1600]

np.random.seed(42)
np.random.shuffle(all_scenarios)

k_folds = 8
n_in_sample = 200

folds = [
    all_scenarios[i * n_in_sample:(i + 1) * n_in_sample]
    for i in range(k_folds)
]

cv_results = []

for fold_idx in range(k_folds):

    train_scenarios = folds[fold_idx]

    test_scenarios = [
        sc
        for idx, fold in enumerate(folds)
        if idx != fold_idx
        for sc in fold
    ]

    # One-price model
    builder_one_cv = DayAheadOnePriceBuilder(
        scenario_list=train_scenarios,
        model_name=f"One-Price CV Fold {fold_idx + 1}"
    )

    problem_one_cv = LP_OptimizationProblem(builder_one_cv)
    problem_one_cv.run()

    p_da_one = [
        problem_one_cv.results.variables[f"p_DA_{hour}"]
        for hour in range(1, builder_one_cv.num_hours + 1)
    ]

    one_price_is = evaluate_one_price_profit(p_da_one, train_scenarios)
    one_price_oos = evaluate_one_price_profit(p_da_one, test_scenarios)

    # Two-price model
    builder_two_cv = DayAheadTwoPriceBuilder(
        scenario_list=train_scenarios,
        model_name=f"Two-Price CV Fold {fold_idx + 1}"
    )

    problem_two_cv = LP_OptimizationProblem(builder_two_cv)
    problem_two_cv.run()

    p_da_two = [
        problem_two_cv.results.variables[f"p_DA_{hour}"]
        for hour in range(1, builder_two_cv.num_hours + 1)
    ]

    two_price_is = evaluate_two_price_profit(p_da_two, train_scenarios)
    two_price_oos = evaluate_two_price_profit(p_da_two, test_scenarios)

    result = {
        "fold": fold_idx + 1,
        "n_in_sample": len(train_scenarios),
        "n_out_of_sample": len(test_scenarios),

        "one_price_expected_profit_is": float(np.mean(one_price_is)),
        "one_price_expected_profit_oos": float(np.mean(one_price_oos)),

        "two_price_expected_profit_is": float(np.mean(two_price_is)),
        "two_price_expected_profit_oos": float(np.mean(two_price_oos)),
    }

    cv_results.append(result)

    print("=" * 70)
    print(f"Fold {fold_idx + 1}")
    print(f"One-price IS:  {result['one_price_expected_profit_is']:.2f}")
    print(f"One-price OOS: {result['one_price_expected_profit_oos']:.2f}")
    print(f"Two-price IS:  {result['two_price_expected_profit_is']:.2f}")
    print(f"Two-price OOS: {result['two_price_expected_profit_oos']:.2f}")
    print("=" * 70)
    
    
    # %% 