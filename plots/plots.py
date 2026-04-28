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



def plot_fcr_profiles_and_bid(problem, builder, num_scenarios_to_plot=20, save_path: Optional[Path] = None):
    """Plot a subset of FCR-D UP scenarios against the optimal bid."""
    c_up_opt = problem.results.variables.get('c_up', 0.0)
    
    # just a few scenarios for better visualization
    num_scenarios = min(num_scenarios_to_plot, builder.num_scenarios)
    f_up_subset = builder.f_up_in[:, :num_scenarios]
    
    
    df_list = []
    for w in range(num_scenarios):
        for m in range(builder.num_minutes):
            df_list.append({
                "Minute": m + 1,
                "Scenario": f"Scenario {w+1}",
                "F_up [MW]": float(f_up_subset[m, w])
            })
    
    df = pd.DataFrame(df_list)
    
    # line plot w
    fig = px.line(
        df, 
        x="Minute", 
        y="F_up [MW]", 
        color="Scenario",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # optimal bid line
    fig.add_hline(
        y=c_up_opt, 
        line_dash="dash", 
        line_color="red", 
        line_width=3,
        annotation_text=f"Optimal Bid (c_up): {c_up_opt:.2f} MW", 
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        xaxis_title="Minute",
        yaxis_title="Reserve Requirement [MW]",
        showlegend=False, 
        template="plotly_white",
        margin=dict(l=5, r=5, t=10, b=10),
        width=800,
        height=400,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))

    return fig


def plot_cvar_shortfall_distribution(problem, builder, save_path: Optional[Path] = None):
    """Plot the distribution of all reserve requirements and the optimal bid."""
    c_up_opt = problem.results.variables.get('c_up', 0.0)
    
    # flatten the entire f_up matrix for histogram
    f_up_flat = builder.f_up_in.flatten()
    
    df = pd.DataFrame({
        "F_up [MW]": f_up_flat
    })
    
    fig = px.histogram(
        df, 
        x="F_up [MW]", 
        nbins=100,
        histnorm="percent",
        color_discrete_sequence=["#636EFA"]
    )
    
    # optimal bid line
    fig.add_vline(
        x=c_up_opt, 
        line_dash="dash", 
        line_color="red", 
        line_width=3
    )
    
    # Annotation
    fig.add_annotation(
        x=c_up_opt,
        y=0.95,
        yref="paper",
        text=f"Optimal Bid: {c_up_opt:.2f} MW<br>Risk region is to the right",
        showarrow=False,
        xanchor="left",
        xshift=10,
        font=dict(color="red"),
    )
    
    fig.update_layout(
        xaxis_title="Reserve Requirement F_up [MW]",
        yaxis_title="Probability [%]",
        template="plotly_white",
        margin=dict(l=5, r=5, t=10, b=10),
        width=800,
        height=400,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))

    return fig



def plot_alsox_discarded_scenarios(problem, builder, save_path: Optional[Path] = None):
    """Plot all FCR-D UP requirements, highlighting the ones discarded by ALSO-X."""
    c_up_opt = problem.results.variables.get('c_up', 0.0)
    
    df_list = []
    
    # extract y_{m,w} values 
    for m in range(builder.num_minutes):
        for w in range(builder.num_scenarios):
            y_val = problem.results.variables.get(f"y_m{m+1}_w{w+1}", 0.0)
            is_discarded = y_val > 0.5 
            
            df_list.append({
                "Minute": m + 1,
                "Scenario": w + 1,
                "F_up [MW]": float(builder.f_up_in[m, w]),
                "Status": "Discarded (y=1)" if is_discarded else "Kept (y=0)"
            })
            
    df = pd.DataFrame(df_list)
    
    # Creiamo lo scatter plot
    fig = px.scatter(
        df, 
        x="Minute", 
        y="F_up [MW]", 
        color="Status",
        color_discrete_map={
            "Discarded (y=1)": "#EF553B", 
            "Kept (y=0)": "#636EFA"      
        },
        opacity=0.6,
        hover_data=["Scenario"]
    )
    
    # optimal bid line
    fig.add_hline(
        y=c_up_opt, 
        line_dash="dash", 
        line_color="black", 
        line_width=2,
        annotation_text=f"Optimal Bid (c_up): {c_up_opt:.2f} MW", 
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="ALSO-X: Kept vs Discarded Reserve Requirements",
        xaxis_title="Minute",
        yaxis_title="Reserve Requirement F_up [MW]",
        template="plotly_white",
        margin=dict(l=5, r=5, t=40, b=10),
        width=900,
        height=500,
        legend_title="Scenario Status"
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))

    return fig

