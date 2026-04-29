import numpy as np
import pandas as pd
import plotly.express as px
import pathlib
from typing import Optional

def generate_fcr_profiles(n_minutes=60, n_scenarios=300, p_min=220, p_max=600, max_delta=35, seed=42):
    """
    Generate stochastic load profiles for Step 2: FCR-D UP.
    Returns a dictionary containing consumption profiles and upward flexibility 
    (split into in-sample and out-of-sample sets).
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize the matrix for load profiles (60 minutes x 300 scenarios)
    profiles = np.zeros((n_minutes, n_scenarios))
    
    #  Generate the load for the very first minute (m=0) across all scenarios
    profiles[0, :] = np.random.uniform(p_min, p_max, size=n_scenarios)
    
    # Generate subsequent minutes using a bounded random walk
    for m in range(1, n_minutes):
        delta = np.random.uniform(-max_delta, max_delta, size=n_scenarios)
        next_val = profiles[m-1, :] + delta
        profiles[m, :] = np.clip(next_val, p_min, p_max)
        
    # Calculate upward flexibility (F_up)
    f_up = profiles 
    
    # Return the full matrices 
    data = {
        "f_up": f_up,            # Shape: 60x300 - Full flexibility matrix
        "profiles": profiles     # Shape: 60x300 - Full load matrix
    }
    
    return data


def plot_fcr_flexibility(f_up_data, num_scenarios: int = 10, save_path: Optional[pathlib.Path] = None):
    """Plot a sample of the FCR-D UP flexibility scenarios with Plotly."""
    # Take only the first 'num_scenarios' for better readability
    sample_data = f_up_data[:, :num_scenarios]
    
    # Convert to DataFrame for Plotly
    df_plot = pd.DataFrame(sample_data, columns=[f"Scenario {i+1}" for i in range(num_scenarios)])
    df_plot.index.name = "Minute"
    df_plot = df_plot.reset_index()

    # Create the Plotly figure
    fig = px.line(
        df_plot,
        x="Minute",
        y=[col for col in df_plot.columns if col.startswith("Scenario")],
        markers=False,
    )
    
    # Apply styling consistent with Step 1
    fig.update_layout(
        title=f"FCR-D UP Flexibility (First {num_scenarios} Scenarios)",
        xaxis_title="Minute",
        yaxis_title="Available Flexibility [kW]",
        template="plotly_white",
        margin=dict(l=25, r=10, t=45, b=25),
        width=1100,
        height=550,
        legend_title="Scenarios"
    )

    return fig


if __name__ == "__main__":
    # Generate the data (all 300 scenarios)
    data = generate_fcr_profiles()
    
    # Define output paths relative to this script
    base_dir = pathlib.Path(__file__).resolve().parent
    csv_path = base_dir / "fcr_flexibility_profiles.csv"
    
    # Plot the first 10 scenarios and save the PDF
    fig = plot_fcr_flexibility(data["f_up"], num_scenarios=10)
    fig.show()  
    
    # Generate and save the CSV
    # Create generic column names for all 300 scenarios
    columns = [f"Scenario_{i}" for i in range(300)]
    
    df_f_up = pd.DataFrame(data["f_up"], columns=columns)
    df_f_up.index.name = "Minute"
    df_f_up.to_csv(csv_path)
    
    print(f"CSV saved at:  {csv_path}")