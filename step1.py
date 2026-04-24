#%%
from data.data import scenarios
from models.StepOne import DayAheadOnePriceBuilder, DayAheadTwoPriceBuilder
from models.OptimizationClasses import LP_OptimizationProblem
from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#%%
scenarios
# %%
builder = DayAheadOnePriceBuilder()
builder.build_objective_coefficients()

# %% Solve optimization problem
problem = LP_OptimizationProblem(builder)
problem.run()
# %%


def plot_optimal_day_ahead_offers(problem, save_path: Optional[Path] = None):
	"""Plot the optimal day-ahead offer schedule."""
	offers = pd.DataFrame(
		{
			"Hour": list(range(1, builder.num_hours + 1)),
			"Offer [MW]": [problem.results.variables[f"p_DA_{hour}"] for hour in range(1, builder.num_hours + 1)],
		}
	)

	fig = px.bar(
		offers,
		x="Hour",
		y="Offer [MW]",
		title="Optimal day-ahead offers",
	)
	fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
	fig.update_layout(
		xaxis_title="Hour",
		yaxis_title="Offer [MW]",
		template="plotly_white",
		margin=dict(l=25, r=10, t=50, b=25),
		width=1100,
		height=550,
	)

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.write_html(save_path)

	return fig

# %%
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
problem.model.write(str(output_dir / "step1_1.lp"))
fig = plot_optimal_day_ahead_offers(problem, save_path=output_dir / "optimal_day_ahead_offers.html")
fig.show(renderer="browser")


# %%
builder2 = DayAheadTwoPriceBuilder()
builder2.build_objective_coefficients()
# %%
