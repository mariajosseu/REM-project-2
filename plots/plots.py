import pandas as pd
import plotly.express as px
from zipp import Path
from typing import Optional

def plot_optimal_day_ahead_offers(problem, builder, save_path: Optional[Path] = None):
	"""Plot the optimal day-ahead offer schedule."""
	price_spread = [
		sum(
			float(s.prices[hour]) - (1.25 * float(s.prices[hour]) if int(s.imbalance[hour]) == 1 else 0.85 * float(s.prices[hour]))
			for s in builder.scenario_list
		) / len(builder.scenario_list)
		for hour in range(builder.num_hours)
	]

	offers = pd.DataFrame(
		{
			"Hour": list(range(1, builder.num_hours + 1)),
			"Offer [MW]": [problem.results.variables[f"p_DA_{hour}"] for hour in range(1, builder.num_hours + 1)],
			"E[λ_DA - λ_B]": price_spread,
		}
	)

	fig = px.bar(
		offers,
		x="Hour",
		y="Offer [MW]"
	)
	fig.add_scatter(
		x=offers["Hour"],
		y=offers["E[λ_DA - λ_B]"],
		mode="lines+markers",
		name="E[λ_DA - λ_B]",
		yaxis="y2",
		line=dict(color="#E4572E", width=2),
	)
	fig.update_layout(
		xaxis_title="Hour",
		xaxis=dict(tickmode="linear", dtick=1),
		yaxis_title="Offer [MW]",
		legend=dict(
			orientation="h",
			yanchor="top",
			y=-0.2,
			xanchor="center",
			x=0.5,
		),
		yaxis2=dict(
			title="E[λ_DA - λ_B] [EUR/MWh]",
			overlaying="y",
			side="right",
		),
		template="plotly_white",
		margin=dict(l=5, r=5, t=5, b=85),
		width=700,
		height=350,
	)

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		if save_path.suffix.lower() == ".html":
			fig.write_html(str(save_path))
		else:
			fig.write_image(str(save_path))

	return fig


def plot_in_sample_profit_distribution(problem, builder, save_path: Optional[Path] = None):
	"""Plot in-sample scenario profit distribution with expected profit marker."""
	scenario_profits = []
	variables = problem.results.variables

	for w, scenario in enumerate(builder.scenario_list, start=1):
		profit = 0.0
		for hour in range(1, builder.num_hours + 1):
			da_price = float(scenario.prices[hour - 1])
			p_da = float(variables[f"p_DA_{hour}"])

			profit += da_price * p_da

			delta_key = f"delta_{hour}_{w}"
			balancing_price = 1.25 * da_price if int(scenario.imbalance[hour - 1]) == 1 else 0.85 * da_price
			profit += balancing_price * float(variables[delta_key])

		scenario_profits.append(profit)

	expected_profit_eur = float(problem.results.objective_value)
	expected_profit_100keur = float(problem.results.objective_value) / 100000.0

	profits = pd.DataFrame(
		{
			"Scenario": list(range(1, len(scenario_profits) + 1)),
			"Profit [100 kEUR]": [p / 100000.0 for p in scenario_profits],
		}
	)

	fig = px.histogram(
		profits,
		x="Profit [100 kEUR]",
		nbins=300,
		histnorm="percent",
	)
	fig.add_vline(
		x=expected_profit_100keur,
		line_color="red",
		line_width=2,
		line_dash="dash",
	)
	fig.add_annotation(
		x=expected_profit_100keur,
		y=0.95,
		yref="paper",
		text=f"Expected profit: {expected_profit_eur:,.0f} EUR",
		showarrow=False,
		xanchor="left",
		xshift=8,
		font=dict(color="red"),
	)
	fig.update_layout(
		xaxis_title="Profit [100 kEUR]",
		xaxis=dict(range=[min(profits["Profit [100 kEUR]"]), 3]),
		yaxis_title="Probability [%]",
		template="plotly_white",
		margin=dict(l=5, r=5, t=5, b=85),
		width=900,
		height=450,
	)

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		if save_path.suffix.lower() == ".html":
			fig.write_html(str(save_path))
		else:
			fig.write_image(str(save_path))

	return fig

