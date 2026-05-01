#%%
from pathlib import Path
from models.StepTwo import CVaRBuilder, ALSOXBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from plots.plots import plot_fcr_spaghetti_violations, plot_bottleneck_cdf
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
    save_path=output_dir / "step2_spaghetti_violations.html"
)

# Generate CDF plot
fig_cdf = plot_bottleneck_cdf(
    f_up_in=f_up_in_matrix, 
    problem_cvar=problem_cvar, 
    problem_alsox=problem_alsox, 
    save_path=output_dir / "step2_bottleneck_cdf.html"
)

# Show in browser
fig_spaghetti.show(renderer="browser")
fig_cdf.show(renderer="browser")
# %%