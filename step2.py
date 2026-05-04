#%%
from pathlib import Path
import numpy as np
import pandas as pd
from models.StepTwo import CVaRBuilder, ALSOXBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from plots.plots import (
    plot_fcr_spaghetti_violations,
    plot_bottleneck_cdf,
    plot_alsox_bid_vs_requirement,
    plot_alsox_shortfall_vs_requirement,
)
#%% 
builder_cvar = CVaRBuilder(alpha=0.90)

# %% Solve optimization problem
problem_cvar = LP_OptimizationProblem(builder_cvar)
problem_cvar.run()

# %% Print Results
c_up_opt = problem_cvar.results.variables.get('c_up', 0.0)
print(f"Optimal Bid (CVaR): {c_up_opt:.2f} kW")

# %% Save Model and Outputs
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Save the optimization model in LP format for inspection
problem_cvar.model.write(str(output_dir / "step2_cvar.lp"))

# %% ALSOX Model (future setup)
builder_alsox = ALSOXBuilder(alpha=0.9)
problem_alsox = LP_OptimizationProblem(builder_alsox)
problem_alsox.run()

# Print ALSO-X Results
c_up_opt_alsox = problem_alsox.results.variables.get('c_up', 0.0)
print(f"Optimal Bid (ALSO-X): {c_up_opt_alsox:.2f} kW")

problem_alsox.model.write(str(output_dir / "step2_alsox.lp"))

# %% Generate and save JOINT plots for CVaR and ALSO-X
# Extract the in-sample flexibility matrix from one of the builders
f_up_in_matrix = builder_cvar.f_up_in

# Generate Spaghetti plot
fig_spaghetti = plot_fcr_spaghetti_violations(
    f_up_in=f_up_in_matrix, 
    problem_cvar=problem_cvar, 
    problem_alsox=problem_alsox, 
    save_path=output_dir / "step2_spaghetti_violations.pdf"
)

# Generate CDF plot
fig_cdf = plot_bottleneck_cdf(
    f_up_in=f_up_in_matrix, 
    problem_cvar=problem_cvar, 
    problem_alsox=problem_alsox, 
    save_path=output_dir / "step2_bottleneck_cdf.pdf"
)

# Show in browser
fig_spaghetti.show(renderer="browser")
fig_cdf.show(renderer="browser")

# %% Task 2.2 - Out-of-sample P90 verification (200 holdout profiles)
csv_path = Path(__file__).resolve().parent / "data" / "fcr_flexibility_profiles.csv"
f_up_all = pd.read_csv(csv_path, index_col=0).values

# Training uses the first 100 profiles in the optimization builders; use the remaining 200 as holdout.
f_up_oos = f_up_all[:, 100:300]

if f_up_oos.shape[1] != 200:
    raise ValueError(f"Expected 200 out-of-sample profiles, got {f_up_oos.shape[1]}")

def verify_p90_average_minute_level(bid_kw: float, f_up_matrix, epsilon: float = 0.10):
    """
    Verify P90 under the average minute-level interpretation.

    A bid satisfies P90 if it is available in at least 90% of all
    minute-profile observations, i.e. if the total shortfall rate over
    all (minute, profile) pairs is <= epsilon.
    """
    shortfall_mask = f_up_matrix < bid_kw

    # Average/sum interpretation: count all violated minute-profile pairs.
    minute_shortfall_rate = shortfall_mask.mean()
    minute_coverage = 1.0 - minute_shortfall_rate

    # Keep profile-level metrics only as diagnostics.
    profile_shortfall_mask = shortfall_mask.any(axis=0)
    profile_shortfall_rate = profile_shortfall_mask.mean()
    profile_coverage = 1.0 - profile_shortfall_rate

    p90_met = minute_shortfall_rate <= epsilon

    return {
        "minute_shortfall_rate": float(minute_shortfall_rate),
        "minute_coverage": float(minute_coverage),
        "profile_shortfall_rate": float(profile_shortfall_rate),
        "profile_coverage": float(profile_coverage),
        "p90_met": bool(p90_met),
    }

verification_cvar = verify_p90_average_minute_level(c_up_opt, f_up_oos)
verification_alsox = verify_p90_average_minute_level(c_up_opt_alsox, f_up_oos)

print("\n" + "=" * 72)
print("TASK 2.2 - P90 Verification with 200 Out-of-Sample Profiles")
print("=" * 72)

print(f"CVaR bid: {c_up_opt:.2f} kW")
print(f"  Profile shortfall rate: {verification_cvar['profile_shortfall_rate']:.2%}")
print(f"  Profile coverage:       {verification_cvar['profile_coverage']:.2%}")
print(f"  Minute shortfall rate:  {verification_cvar['minute_shortfall_rate']:.2%}")
print(f"  Minute coverage:        {verification_cvar['minute_coverage']:.2%}")
print(f"  P90 requirement met:    {verification_cvar['p90_met']}")

print(f"ALSO-X bid: {c_up_opt_alsox:.2f} kW")
print(f"  Profile shortfall rate: {verification_alsox['profile_shortfall_rate']:.2%}")
print(f"  Profile coverage:       {verification_alsox['profile_coverage']:.2%}")
print(f"  Minute shortfall rate:  {verification_alsox['minute_shortfall_rate']:.2%}")
print(f"  Minute coverage:        {verification_alsox['minute_coverage']:.2%}")
print(f"  P90 requirement met:    {verification_alsox['p90_met']}")

# %% Task 2.3 - Energinet perspective (ALSO-X sensitivity to P-threshold)
def compute_shortfall_metrics(bid_kw: float, f_up_matrix: np.ndarray):
    """Compute out-of-sample shortfall metrics for a fixed reserve bid."""
    shortfall_kw = np.maximum(bid_kw - f_up_matrix, 0.0)
    shortfall_mask = shortfall_kw > 0

    expected_shortfall_kw = float(shortfall_kw.mean())
    shortfall_probability = float(shortfall_mask.mean())

    if shortfall_mask.any():
        avg_shortfall_given_shortfall_kw = float(shortfall_kw[shortfall_mask].mean())
    else:
        avg_shortfall_given_shortfall_kw = 0.0

    return {
        "expected_shortfall_kw": expected_shortfall_kw,
        "shortfall_probability": shortfall_probability,
        "avg_shortfall_given_shortfall_kw": avg_shortfall_given_shortfall_kw,
    }


alpha_grid = np.linspace(0.80, 1.00, 11)
task23_rows = []

for alpha in alpha_grid:
    builder_alsox_grid = ALSOXBuilder(alpha=float(alpha), model_name=f"ALSOX_P{int(round(alpha * 100))}")
    problem_alsox_grid = LP_OptimizationProblem(builder_alsox_grid)
    problem_alsox_grid.run()

    bid_kw = float(problem_alsox_grid.results.variables.get("c_up", 0.0))
    metrics = compute_shortfall_metrics(bid_kw, f_up_oos)

    epsilon = max(0.0, 1.0 - float(alpha))
    tol = 1e-9
    task23_rows.append(
        {
            "P_requirement": f"P{int(round(alpha * 100))}",
            "alpha": float(alpha),
            "epsilon_allowed_shortfall": epsilon,
            "alsox_bid_in_sample_kw": bid_kw,
            "oos_expected_shortfall_kw": metrics["expected_shortfall_kw"],
            "oos_shortfall_probability": metrics["shortfall_probability"],
            "oos_avg_shortfall_given_shortfall_kw": metrics["avg_shortfall_given_shortfall_kw"],
            "oos_requirement_met": metrics["shortfall_probability"] <= epsilon + tol,
        }
    )

task23_df = pd.DataFrame(task23_rows)
task23_df = task23_df.sort_values("alpha").reset_index(drop=True)
task23_df.to_csv(output_dir / "task2_3_alsox_sensitivity.csv", index=False)

print("\n" + "=" * 72)
print("TASK 2.3 - ALSO-X Sensitivity (P80 to P100)")
print("=" * 72)
print(
    task23_df[
        [
            "P_requirement",
            "alsox_bid_in_sample_kw",
            "oos_expected_shortfall_kw",
            "oos_shortfall_probability",
            "oos_requirement_met",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

bid_change = task23_df.iloc[-1]["alsox_bid_in_sample_kw"] - task23_df.iloc[0]["alsox_bid_in_sample_kw"]
shortfall_change = task23_df.iloc[-1]["oos_expected_shortfall_kw"] - task23_df.iloc[0]["oos_expected_shortfall_kw"]

print("\nTrade-off summary (P80 -> P100):")
print(f"  In-sample optimal bid change: {bid_change:.2f} kW")
print(f"  OOS expected shortfall change: {shortfall_change:.2f} kW")
print(f"  Results saved to: {output_dir / 'task2_3_alsox_sensitivity.csv'}")
# %%

fig_bid_sensitivity = plot_alsox_bid_vs_requirement(
    task23_df,
    save_path=output_dir / "task2_3_alsox_bid_vs_requirement.pdf",
)

fig_shortfall_sensitivity = plot_alsox_shortfall_vs_requirement(
    task23_df,
    save_path=output_dir / "task2_3_alsox_shortfall_vs_requirement.pdf",
)

fig_bid_sensitivity.show(renderer="browser")
fig_shortfall_sensitivity.show(renderer="browser")

# %%
