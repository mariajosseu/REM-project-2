def compute_one_price_profits(problem, builder):
        scenario_profits = []
        variables = problem.results.variables

        for w, scenario in enumerate(builder.scenario_list, start=1):
            profit = 0.0

            for hour in range(1, builder.num_hours + 1):
                da_price = float(scenario.prices[hour - 1])
                p_da = float(variables[f"p_DA_{hour}"])
                delta = float(variables[f"delta_{hour}_{w}"])

                balancing_price = (
                    1.25 * da_price
                    if int(scenario.imbalance[hour - 1]) == 1
                    else 0.85 * da_price
                )

                profit += da_price * p_da
                profit += balancing_price * delta

            scenario_profits.append(profit)

        return scenario_profits

def compute_two_price_profits(problem, builder):
        scenario_profits = []
        variables = problem.results.variables

        for w, scenario in enumerate(builder.scenario_list, start=1):
            profit = 0.0

            for hour in range(1, builder.num_hours + 1):
                da_price = float(scenario.prices[hour - 1])
                p_da = float(variables[f"p_DA_{hour}"])
                delta_up = float(variables[f"delta_up_{hour}_{w}"])
                delta_down = float(variables[f"delta_down_{hour}_{w}"])

                if int(scenario.imbalance[hour - 1]) == 1:
                    # System deficit:
                    # Upward deviation helps -> settled at DA
                    # Downward deviation hurts -> settled at 1.25 * DA
                    profit += da_price * p_da
                    profit += da_price * delta_up
                    profit -= 1.25 * da_price * delta_down

                else:
                    # System surplus:
                    # Upward deviation hurts -> settled at 0.85 * DA
                    # Downward deviation helps -> settled at DA
                    profit += da_price * p_da
                    profit += 0.85 * da_price * delta_up
                    profit -= da_price * delta_down

            scenario_profits.append(profit)

        return scenario_profits

def evaluate_one_price_profit(p_da, scenarios_to_evaluate):
    """Evaluate fixed day-ahead offers under one-price settlement."""
    scenario_profits = []

    for scenario in scenarios_to_evaluate:
        profit = 0.0

        for hour in range(1, 25):
            da_price = float(scenario.prices[hour - 1])
            wind = float(scenario.wind[hour - 1])
            offer = float(p_da[hour - 1])

            delta = wind - offer

            balancing_price = (
                1.25 * da_price
                if int(scenario.imbalance[hour - 1]) == 1
                else 0.85 * da_price
            )

            profit += da_price * offer
            profit += balancing_price * delta

        scenario_profits.append(profit)

    return scenario_profits


def evaluate_two_price_profit(p_da, scenarios_to_evaluate):
    """Evaluate fixed day-ahead offers under two-price settlement."""
    scenario_profits = []

    for scenario in scenarios_to_evaluate:
        profit = 0.0

        for hour in range(1, 25):
            da_price = float(scenario.prices[hour - 1])
            wind = float(scenario.wind[hour - 1])
            offer = float(p_da[hour - 1])

            delta = wind - offer
            delta_up = max(delta, 0.0)
            delta_down = max(-delta, 0.0)

            profit += da_price * offer

            if int(scenario.imbalance[hour - 1]) == 1:
                # System deficit:
                # upward deviation helps -> DA price
                # downward deviation hurts -> 1.25 * DA price
                profit += da_price * delta_up
                profit -= 1.25 * da_price * delta_down
            else:
                # System surplus:
                # upward deviation hurts -> 0.85 * DA price
                # downward deviation helps -> DA price
                profit += 0.85 * da_price * delta_up
                profit -= da_price * delta_down

        scenario_profits.append(profit)

    return scenario_profits