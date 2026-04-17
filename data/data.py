import pandas as pd
import pathlib
import numpy as np
import itertools
from dataclasses import dataclass
from typing import Optional
import plotly.express as px

# Wind power generation data (Fingrid)
def load_wind_forecasts():
    """Load 15 - minute wind power generation"""
    file_path = pathlib.Path(__file__).parent / f'75_2026-03-01T0100_2026-03-21T0100.csv'
    df = pd.read_csv(file_path, sep=';', parse_dates=['startTime'])
    df = df.set_index('startTime')
    df_hourly = df.resample('h').mean(numeric_only=True)

    return df_hourly


# Electricity price data (Nordpool, Finland)
def load_electricity_prices():
    """Load hourly electricity prices and reshape to long format"""
    file_path = pathlib.Path(__file__).parent / f'hourly_avg_prices_march_2025.csv'
    df = pd.read_csv(file_path)
    date_columns = [col for col in df.columns if col != 'Hour']
    df_long = df.melt(
        id_vars="Hour",
        value_vars=date_columns,
        var_name="Date",
        value_name="Price"
    )
    df_long["Date"] = pd.to_datetime(df_long["Date"])
    df_long["idx"] = df_long["Date"].astype(str) + "_" + df_long["Hour"]
    df_single_idx = df_long.set_index("idx").sort_index()

    return df_single_idx



# Bernoulli distribution for imbalance modeling
def load_imbalance_data():
    imbalance_data = np.random.binomial(n=1, p=0.5, size=(4, 24))
    return imbalance_data

wind_forecast = load_wind_forecasts()
imbalance_data = load_imbalance_data()
electricity_prices = load_electricity_prices()


@dataclass
class Scenario:
    wind_day:      int          # indice giorno wind (0-19)
    price_day:     int          # indice giorno price (0-19)
    imbalance_day: int          # indice giorno imbalance (0-3)
    wind:          np.ndarray   # shape (24,) — produzione oraria [MW]
    prices:        np.ndarray   # shape (24,) — prezzi orari [€/MWh]
    imbalance:     np.ndarray   # shape (24,) — flag imbalance {0,1}


def build_scenarios(wind_forecast, electricity_prices, imbalance_data) -> dict:
    """ Returns a dict keyed by (wind_day, price_day, imbalance_day) -> Scenario
        Total: 20 * 20 * 4 = 1600 entries
    """
    # --- 1. Slice wind into 20 daily arrays (24h each) ---
    wind_days = [
        wind_forecast.iloc[d * 24 : (d + 1) * 24]["Wind power generation - 15 min data"].values 
        for d in range(20)
    ]

    # --- 2. Slice prices into 20 daily arrays ---
    unique_dates = sorted(electricity_prices["Date"].unique())[:20]
    price_days = [
        electricity_prices[electricity_prices["Date"] == date]["Price"].values
        for date in unique_dates
    ]

    # --- 3. Imbalance: 4 rows already present (shape 4×24) ---
    imbalance_days = [imbalance_data[i] for i in range(4)]

    # --- 4. Cartesian product → 1600 scenarios ---
    scenarios: dict[tuple, Scenario] = {}

    for w, p, b in itertools.product(range(20), range(20), range(4)):
        scaled_prices = np.where(
            imbalance_days[b] == 1,
            price_days[p] * 1.25,
            price_days[p] * 0.85,
        )
        scenarios[(w, p, b)] = Scenario(
            wind_day      = w,
            price_day     = p,
            imbalance_day = b,
            wind          = wind_days[w],
            prices        = scaled_prices,
            imbalance     = imbalance_days[b],
        )

    return scenarios


def plot_wind_sample(wind_data, sample_hours: int = 48, save_path: Optional[pathlib.Path] = None):
    """Plot a sample of the hourly wind forecast data with Plotly."""
    sample = wind_data.head(sample_hours).reset_index()

    fig = px.line(
        sample,
        x="startTime",
        y="Wind power generation - 15 min data",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Wind power [MW]",
        template="plotly_white",
        margin=dict(l=25, r=10, t=45, b=25),
        width=1100,
        height=550,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_path)

    return fig


scenarios = build_scenarios(wind_forecast, electricity_prices, imbalance_data)


if __name__ == "__main__":
    output_path = pathlib.Path(__file__).resolve().parent.parent / "plots" / "wind_forecast_full.pdf"
    fig = plot_wind_sample(wind_forecast, sample_hours=len(wind_forecast), save_path=output_path)
