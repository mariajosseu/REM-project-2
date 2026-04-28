from pathlib import Path
from models.StepTwo import CVaRBuilder, ALSOXBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from plots.plots import plot_fcr_profiles_and_bid, plot_cvar_shortfall_distribution, plot_alsox_discarded_scenarios
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


# Genera e salva i plot
fig1 = plot_fcr_profiles_and_bid(problem_cvar, builder_cvar, num_scenarios_to_plot=30, save_path=output_dir / "cvar_profiles_vs_bid.html")
fig2 = plot_cvar_shortfall_distribution(problem_cvar, builder_cvar, save_path=output_dir / "cvar_shortfall_distribution.html")

# Mostra nel browser (opzionale, utile per i test interattivi)
fig1.show(renderer="browser")
fig2.show(renderer="browser")

# %% ALSOX Model (Setup futuro)
builder_alsox = ALSOXBuilder(epsilon=0.1)
problem_alsox = LP_OptimizationProblem(builder_alsox)
problem_alsox.run()

# Print ALSO-X Results
c_up_opt_alsox = problem_alsox.results.variables.get('c_up', 0.0)
print(f"Optimal Bid (ALSO-X): {c_up_opt_alsox:.2f} kW")

problem_alsox.model.write(str(output_dir / "step2_alsox.lp"))

# Genera e salva il plot per ALSO-X
fig3 = plot_alsox_discarded_scenarios(problem_alsox, builder_alsox, save_path=output_dir / "alsox_discarded_scenarios.html")

# Mostra nel browser
fig3.show(renderer="browser")
# %%