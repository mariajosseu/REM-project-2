import pandas as pd
import plotly.express as px
from zipp import Path
from typing import Optional
import numpy as np

from utils import compute_one_price_profits, compute_two_price_profits

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


def plot_optimal_day_ahead_offers_with_avg_imbalance(problem, builder, save_path: Optional[Path] = None):
	"""Plot the optimal day-ahead offer schedule with average imbalance per hour."""
	avg_imbalance = [
		sum(float(s.imbalance[hour]) for s in builder.scenario_list) / len(builder.scenario_list)
		for hour in range(builder.num_hours)
	]

	offers = pd.DataFrame(
		{
			"Hour": list(range(1, builder.num_hours + 1)),
			"Offer [MW]": [problem.results.variables[f"p_DA_{hour}"] for hour in range(1, builder.num_hours + 1)],
			"Avg imbalance": avg_imbalance,
		}
	)

	fig = px.bar(
		offers,
		x="Hour",
		y="Offer [MW]"
	)
	fig.add_scatter(
		x=offers["Hour"],
		y=offers["Avg imbalance"],
		mode="lines+markers",
		name="Avg imbalance",
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
			title="Avg imbalance [0/1]",
			overlaying="y",
			side="right",
			range=[0, 1],
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

def plot_one_price_vs_two_price_offers(problem_one, problem_two, builder, save_path: Optional[Path] = None):
    """Plot and compare optimal day-ahead offers under one-price and two-price schemes."""

    offers = pd.DataFrame(
        {
            "Hour": list(range(1, builder.num_hours + 1)),
            "One-price offer [MW]": [
                problem_one.results.variables[f"p_DA_{hour}"]
                for hour in range(1, builder.num_hours + 1)
            ],
            "Two-price offer [MW]": [
                problem_two.results.variables[f"p_DA_{hour}"]
                for hour in range(1, builder.num_hours + 1)
            ],
        }
    )

    offers_long = offers.melt(
        id_vars="Hour",
        value_vars=["One-price offer [MW]", "Two-price offer [MW]"],
        var_name="Balancing scheme",
        value_name="Offer [MW]",
    )

    fig = px.bar(
        offers_long,
        x="Hour",
        y="Offer [MW]",
        color="Balancing scheme",
        barmode="group",
    )

    fig.update_layout(
        xaxis_title="Hour",
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis_title="Offer [MW]",
        yaxis=dict(range=[0, builder.P_max * 1.1]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
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


def plot_profit_distribution_comparison(
    problem_one,
    builder_one,
    problem_two,
    builder_two,
    save_path: Optional[Path] = None,
):
    """Compare in-sample profit distributions for one-price and two-price schemes."""

    one_price_profits = compute_one_price_profits(problem_one, builder_one)
    two_price_profits = compute_two_price_profits(problem_two, builder_two)

    profits = pd.DataFrame(
        {
            "Profit [100 kEUR]": (
                [p / 100000.0 for p in one_price_profits]
                + [p / 100000.0 for p in two_price_profits]
            ),
            "Balancing scheme": (
                ["One-price"] * len(one_price_profits)
                + ["Two-price"] * len(two_price_profits)
            ),
        }
    )

    fig = px.histogram(
        profits,
        x="Profit [100 kEUR]",
        color="Balancing scheme",
        nbins=60,
        histnorm="percent",
        barmode="overlay",
        opacity=0.60,
    )

    expected_one = float(problem_one.results.objective_value) / 100000.0
    expected_two = float(problem_two.results.objective_value) / 100000.0

    fig.add_vline(
        x=expected_one,
        line_dash="dash",
        line_width=2,
        line_color="#1f77b4",
    )

    fig.add_vline(
        x=expected_two,
        line_dash="dot",
        line_width=2,
        line_color="#ff7f0e",
    )

    labels_are_close = abs(expected_one - expected_two) < 0.12
    y_one = 0.97
    y_two = 0.88 if labels_are_close else 0.97

    fig.add_annotation(
        x=expected_one,
        y=y_one,
        yref="paper",
        text=f"Expected one-price: {expected_one:.2f}",
        showarrow=False,
        xanchor="left",
        xshift=8,
        font=dict(color="#1f77b4"),
        bgcolor="rgba(255,255,255,0.85)",
    )

    fig.add_annotation(
        x=expected_two,
        y=y_two,
        yref="paper",
        text=f"Expected two-price: {expected_two:.2f}",
        showarrow=False,
        xanchor="right",
        xshift=-8,
        font=dict(color="#ff7f0e"),
        bgcolor="rgba(255,255,255,0.85)",
    )

    fig.update_layout(
        xaxis_title="Profit [100 kEUR]",
        yaxis_title="Probability [%]",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
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
    """Plot all flexibility trajectories and highlight scenarios violating the ALSO-X bid."""
    c_up_cvar = problem_cvar.results.variables["c_up"]
    c_up_alsox = problem_alsox.results.variables["c_up"]

    num_minutes, num_scenarios = f_up_in.shape

    fig = go.Figure()

    violated_scenarios = []
    for w in range(num_scenarios):
        min_flex = np.min(f_up_in[:, w])
        if min_flex < c_up_alsox - 1e-4:
            violated_scenarios.append(w)

    for w in range(num_scenarios):
        if w not in violated_scenarios:
            fig.add_trace(go.Scatter(
                x=list(range(num_minutes)),
                y=f_up_in[:, w],
                mode="lines",
                line=dict(color="rgba(142, 202, 230, 0.3)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

    for i, w in enumerate(violated_scenarios):
        fig.add_trace(go.Scatter(
            x=list(range(num_minutes)),
            y=f_up_in[:, w],
            mode="lines",
            line=dict(color="rgba(228, 87, 46, 0.8)", width=1.5),
            name="Violated Scenarios (P90)" if i == 0 else None,
            showlegend=True if i == 0 else False,
        ))

    fig.add_hline(
        y=c_up_alsox,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text=f"ALSO-X Bid ({c_up_alsox:.1f} kW)",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=c_up_cvar,
        line_dash="dot",
        line_color="green",
        line_width=2,
        annotation_text=f"CVaR Bid ({c_up_cvar:.1f} kW)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Flexibility Trajectories (Violations: {len(violated_scenarios)}/{num_scenarios})",
        xaxis_title="Minute",
        yaxis_title="Available Flexibility [kW]",
        template="plotly_white",
        margin=dict(l=5, r=5, t=40, b=5),
        width=800,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))

    return fig


def plot_bottleneck_cdf(f_up_in, problem_cvar, problem_alsox, save_path: Optional[Path] = None):
    """Plot the ECDF of per-scenario hourly bottlenecks and compare ALSO-X vs CVaR bids."""
    c_up_cvar = problem_cvar.results.variables["c_up"]
    c_up_alsox = problem_alsox.results.variables["c_up"]

    min_flex_per_scenario = np.min(f_up_in, axis=0)

    fig = px.ecdf(
        x=min_flex_per_scenario,
        markers=True,
        lines=True,
        title="CDF of Hourly Bottlenecks (Minimum Flexibility)",
    )

    fig.update_traces(line_color="#219EBC", marker=dict(size=4))

    fig.add_hline(
        y=0.10,
        line_dash="dash",
        line_color="gray",
        annotation_text="10% Probability Threshold (P90)",
        annotation_position="top left",
    )

    fig.add_vline(
        x=c_up_alsox,
        line_dash="solid",
        line_color="#E4572E",
        annotation_text=f"ALSO-X ({c_up_alsox:.1f} kW)",
        annotation_position="bottom right",
    )

    fig.add_vline(
        x=c_up_cvar,
        line_dash="solid",
        line_color="green",
        annotation_text=f"CVaR ({c_up_cvar:.1f} kW)",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="Minimum Hourly Flexibility [kW]",
        yaxis_title="Cumulative Probability",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        template="plotly_white",
        margin=dict(l=5, r=5, t=40, b=5),
        width=800,
        height=450,
    )

    fig.add_vrect(
        x0=min_flex_per_scenario.min(),
        x1=c_up_alsox,
        fillcolor="red",
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="Tail Risk Area",
        annotation_position="top left",
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))

    return fig

def plot_alsox_bid_vs_requirement(task23_df, save_path: Optional[Path] = None):
    """Plot ALSO-X optimal in-sample bid as a function of the P-requirement."""

    fig = px.line(
        task23_df,
        x="P_requirement",
        y="alsox_bid_in_sample_kw",
        markers=True,
    )

    fig.update_layout(
        xaxis_title="Reliability requirement",
        yaxis_title="Optimal ALSO-X bid [kW]",
        template="plotly_white",
        margin=dict(l=5, r=5, t=30, b=60),
        width=800,
        height=450,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))

    return fig


def plot_alsox_shortfall_vs_requirement(task23_df, save_path: Optional[Path] = None):
    """Plot out-of-sample expected shortfall and shortfall probability as a function of the P-requirement."""

    fig = px.line(
        task23_df,
        x="P_requirement",
        y="oos_expected_shortfall_kw",
        markers=True,
    )

    fig.update_traces(
        name="OOS expected shortfall [kW]",
        showlegend=True
    )

    fig.add_scatter(
        x=task23_df["P_requirement"],
        y=100 * task23_df["oos_shortfall_probability"],
        mode="lines+markers",
        name="OOS shortfall probability [%]",
        yaxis="y2",
    )

    fig.update_layout(
        xaxis_title="Reliability requirement",
        yaxis_title="OOS expected shortfall [kW]",
        yaxis2=dict(
            title="OOS shortfall probability [%]",
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        margin=dict(l=5, r=5, t=30, b=80),
        width=850,
        height=450,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))

    return fig