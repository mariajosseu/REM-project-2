#%%
from data.data import scenarios
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from plots.plots import plot_optimal_day_ahead_offers, plot_in_sample_profit_distribution, plot_profit_distribution_comparison, plot_one_price_vs_two_price_offers, plot_expected_profit_vs_cvar, plot_profit_distributions_by_beta
from models.StepOne import DayAheadOnePriceBuilder, DayAheadTwoPriceBuilder, RiskAverseOnePriceBuilder, RiskAverseTwoPriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from utils import compute_one_price_profits, compute_two_price_profits, evaluate_one_price_profit, evaluate_two_price_profit, compute_cvar


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
    
    
# %% Step 1.4 - Varying beta
beta_results_one_price= []
beta_profits_one_price = {}

for beta in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    builder4 = RiskAverseOnePriceBuilder(scenario_list=all_scenarios, model_name=f"Risk-Averse One-Price Model (beta={beta})", beta=beta)
    builder4.build_objective_coefficients()

    problem4 = LP_OptimizationProblem(builder4)
    problem4.run()
    results = problem4.get_results()

    profits = compute_one_price_profits(problem4, builder4)
    beta_profits_one_price[beta] = profits

    beta_results_one_price.append({
        "beta": beta,
        "expected_profit": np.mean(profits),
        "cvar": compute_cvar(problem4, builder4)
    })

print("Min profits one-price:", min(min(profits) for profits in beta_profits_one_price.values()))
fig = plot_expected_profit_vs_cvar(beta_results_one_price, save_path=output_dir / "expected_profit_vs_cvar_1PriceScheme.pdf")
# %% 
beta_results_two_price = []
beta_profits_two_price = {}

for beta in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    builder5 = RiskAverseTwoPriceBuilder(scenario_list=all_scenarios, model_name=f"Risk-Averse Two-Price Model (beta={beta})", beta=beta)
    builder5.build_objective_coefficients()

    problem5 = LP_OptimizationProblem(builder5)
    problem5.run()
    results = problem5.get_results()

    profits = compute_two_price_profits(problem5, builder5)
    beta_profits_two_price[beta] = profits

    beta_results_two_price.append({
        "beta": beta,
        "expected_profit": np.mean(profits),
        "cvar": compute_cvar(problem5, builder5)
    })

print("Min profits two-price:", min(min(profits) for profits in beta_profits_two_price.values()))

fig2 = plot_expected_profit_vs_cvar(beta_results_two_price, save_path=output_dir / "expected_profit_vs_cvar_2PriceScheme.pdf")
# %%
# Plot profit distribution for selected beta values
fig_dist_one = plot_profit_distributions_by_beta(
    beta_profits_one_price,
    title="One-Price: Profit Distribution by Beta",
    save_path=output_dir / "profit_distribution_by_beta_1price.pdf"
)

fig_dist_two = plot_profit_distributions_by_beta(
    beta_profits_two_price,
    title="Two-Price: Profit Distribution by Beta",
    save_path=output_dir / "profit_distribution_by_beta_2price.pdf"
)

# %% sensitivity analysis

BETAS          = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
SCENARIO_STEPS = list(range(200, 1601, 200))   # [200, 400, …, 1600]
COLORMAP       = px.colors.sequential.Viridis  # one colour per scenario count

records_one   = []
records_two   = []

for n_scen in SCENARIO_STEPS:
    subset = all_scenarios[:n_scen]

    for beta in BETAS:
        # ── One-price ────────────────────────────────────────────────────────
        b1 = RiskAverseOnePriceBuilder(
            scenario_list=subset,
            model_name=f"1P b={beta} n={n_scen}",
            beta=beta,
        )
        p1 = LP_OptimizationProblem(b1)
        p1.run()
        records_one.append({
            "n_scenarios":     n_scen,
            "beta":            beta,
            "expected_profit": np.mean(compute_one_price_profits(p1, b1)),
            "cvar":            compute_cvar(p1, b1),
        })

        # ── Two-price ────────────────────────────────────────────────────────
        b2 = RiskAverseTwoPriceBuilder(
            scenario_list=subset,
            model_name=f"2P b={beta} n={n_scen}",
            beta=beta,
        )
        p2 = LP_OptimizationProblem(b2)
        p2.run()
        records_two.append({
            "n_scenarios":     n_scen,
            "beta":            beta,
            "expected_profit": np.mean(compute_two_price_profits(p2, b2)),
            "cvar":            compute_cvar(p2, b2),
        })

df_one = pd.DataFrame(records_one)
df_two = pd.DataFrame(records_two)


# ── Plotting helper ──────────────────────────────────────────────────────────
def plot_sensitivity(df, title, save_path=None):
    """One trace per scenario-subset, x=CVaR, y=E[profit], coloured by n_scenarios."""
    fig = go.Figure()
    n_vals   = sorted(df["n_scenarios"].unique())
    colors   = px.colors.sample_colorscale("Viridis", [i/(len(n_vals)-1) for i in range(len(n_vals))])

    for color, n in zip(colors, n_vals):
        sub = df[df["n_scenarios"] == n].sort_values("cvar")
        fig.add_trace(go.Scatter(
            x=sub["cvar"]            / 1000,
            y=sub["expected_profit"] / 1000,
            mode="lines+markers+text",
            name=f"{n} scenarios",
            text=[f"β={b}" for b in sub["beta"]],
            textposition="top center",
            line=dict(color=color),
            marker=dict(size=7, color=color),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="CVaR [kEUR]",
        yaxis_title="Expected profit [kEUR]",
        template="plotly_white",
        width=900,
        height=550,
        legend_title="Scenario subset",
        margin=dict(l=5, r=5, t=40, b=85),
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))
    return fig


fig_one = plot_sensitivity(
    df_one,
    title="One-price: Expected profit vs CVaR, sensitivity to in-sample scenarios",
    save_path=output_dir / "sensitivity_1price.pdf",
)

fig_two = plot_sensitivity(
    df_two,
    title="Two-price: Expected profit vs CVaR, sensitivity to in-sample scenarios",
    save_path=output_dir / "sensitivity_2price.pdf",
)
