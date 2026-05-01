#%%
from data.data import scenarios
from models.StepOne import DayAheadOnePriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from pathlib import Path
import numpy as np

#%%
all_scenarios = list(scenarios.values())


k_folds = 8
n_samples = 200


folds = [
    all_scenarios[i * n_samples:(i + 1) * n_samples]
    for i in range(k_folds)
]


def scenario_profit(p_da, scenario, num_hours):
    return sum(
        p_da[t] * float(scenario.prices[t])
        - (float(scenario.wind[t]) - p_da[t]) * float(scenario.prices[t])
        for t in range(num_hours)
    )

# %%
cv_results = []

for fold_idx in range(k_folds):
    train_scenarios = folds[fold_idx]
    test_scenarios = [
        sc for idx, fold in enumerate(folds) if idx != fold_idx for sc in fold
    ]
    # %%
    builder = DayAheadOnePriceBuilder(
        scenario_list=train_scenarios,
        model_name=f"Day-Ahead One-Price Model Fold {fold_idx + 1}"
    )

    problem = LP_OptimizationProblem(builder)
    problem.run()
    results = problem.get_results()

    p_DA = [results["variables"][f"p_DA_{t+1}"] for t in range(builder.num_hours)]
    # %%
    profits_is = [
        scenario_profit(p_DA, sc, builder.num_hours)
        for sc in train_scenarios
    ]
    profits_oos = [
        scenario_profit(p_DA, sc, builder.num_hours)
        for sc in test_scenarios
    ]

    result = {
        "fold": fold_idx + 1,
        "n_in_sample": len(train_scenarios),
        "n_out_of_sample": len(test_scenarios),
        "expected_profit_is": float(np.mean(profits_is)),
        "expected_profit_oos": float(np.mean(profits_oos)),
    }
    cv_results.append(result)
    print("=" * 50)
    print(f"profits for fold {fold_idx + 1}: IS = {result['expected_profit_is']:.2f}, OOS = {result['expected_profit_oos']:.2f}")
    print("=" * 50)
# %%


