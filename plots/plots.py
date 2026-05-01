import pandas as pd
import plotly.express as px
from zipp import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go

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





def plot_fcr_spaghetti_violations(f_up_in, problem_cvar, problem_alsox, save_path: Optional[Path] = None):
    """
    Spaghetti plot of all 100 scenarios over 60 minutes.
    Highlights the scenarios that violate the ALSO-X bid in red, 
    proving visually that exactly ~10% of trajectories breach the threshold.
    """
    c_up_cvar = problem_cvar.results.variables['c_up']
    c_up_alsox = problem_alsox.results.variables['c_up']
    
    num_minutes, num_scenarios = f_up_in.shape
    
    fig = go.Figure()

    # Identify scenarios that violate ALSO-X (drop below the bid in at least one minute)
    violated_scenarios = []
    for w in range(num_scenarios):
        min_flex = np.min(f_up_in[:, w])
        if min_flex < c_up_alsox - 1e-4: # Piccola tolleranza numerica
            violated_scenarios.append(w)

	# Plot safe scenarios first (light gray / light blue)
    for w in range(num_scenarios):
        if w not in violated_scenarios:
            fig.add_trace(go.Scatter(
                x=list(range(num_minutes)), y=f_up_in[:, w],
                mode='lines', line=dict(color='rgba(142, 202, 230, 0.3)', width=1),
                showlegend=False, hoverinfo='skip'
            ))

	# Then plot violated scenarios (bright red) to highlight them
    for i, w in enumerate(violated_scenarios):
        fig.add_trace(go.Scatter(
            x=list(range(num_minutes)), y=f_up_in[:, w],
            mode='lines', line=dict(color='rgba(228, 87, 46, 0.8)', width=1.5),
            name="Violated Scenarios (P90)" if i == 0 else None,
            showlegend=True if i == 0 else False
        ))

	# Add the bid lines
    fig.add_hline(y=c_up_alsox, line_dash="dash", line_color="black", line_width=2,
                  annotation_text=f"ALSO-X Bid ({c_up_alsox:.1f} kW)", annotation_position="bottom right")
    fig.add_hline(y=c_up_cvar, line_dash="dot", line_color="green", line_width=2,
                  annotation_text=f"CVaR Bid ({c_up_cvar:.1f} kW)", annotation_position="bottom right")

    fig.update_layout(
        title=f"Flexibility Trajectories (Violations: {len(violated_scenarios)}/{num_scenarios})",
        xaxis_title="Minute",
        yaxis_title="Available Flexibility [kW]",
        template="plotly_white",
        margin=dict(l=5, r=5, t=40, b=5),
        width=800, height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(save_path)) if save_path.suffix != ".html" else fig.write_html(str(save_path))

    return fig


def plot_bottleneck_cdf(f_up_in, problem_cvar, problem_alsox, save_path: Optional[Path] = None):
    """
    Plots the Empirical Cumulative Distribution Function (CDF) of the MINIMUM hourly flexibility.
    Beautifully illustrates the statistical definition of CVaR vs ALSO-X (Chance Constraint).
    """
    c_up_cvar = problem_cvar.results.variables['c_up']
    c_up_alsox = problem_alsox.results.variables['c_up']
    
    # Find the bottleneck (minimum) for each scenario
    min_flex_per_scenario = np.min(f_up_in, axis=0)
    
    fig = px.ecdf(
        x=min_flex_per_scenario, 
        markers=True, lines=True,
        title="CDF of Hourly Bottlenecks (Minimum Flexibility)"
    )
    
    fig.update_traces(line_color="#219EBC", marker=dict(size=4))

    # Add the 10% threshold (P90)
    fig.add_hline(y=0.10, line_dash="dash", line_color="gray", 
                  annotation_text="10% Probability Threshold (P90)", annotation_position="top left")

    # Add the bids as vertical lines
    fig.add_vline(x=c_up_alsox, line_dash="solid", line_color="#E4572E",
                  annotation_text=f"ALSO-X ({c_up_alsox:.1f} kW)", annotation_position="bottom right")
    
    fig.add_vline(x=c_up_cvar, line_dash="solid", line_color="green",
                  annotation_text=f"CVaR ({c_up_cvar:.1f} kW)", annotation_position="top left")

    fig.update_layout(
        xaxis_title="Minimum Hourly Flexibility [kW]",
        yaxis_title="Cumulative Probability",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        template="plotly_white",
        margin=dict(l=5, r=5, t=40, b=5),
        width=800, height=450,
    )

    # Shade the "Tail Risk" area handled by CVaR (points left of ALSO-X)
    fig.add_vrect(x0=min_flex_per_scenario.min(), x1=c_up_alsox, 
                  fillcolor="red", opacity=0.1, layer="below", line_width=0,
                  annotation_text="Tail Risk Area", annotation_position="top left")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(save_path)) if save_path.suffix != ".html" else fig.write_html(str(save_path))

    return fig

