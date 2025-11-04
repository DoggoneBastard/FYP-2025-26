import plotly.graph_objects as go
import plotly.io as pio

# Data for the comparison table
characteristics = [
    "Type",
    "Complexity", 
    "Training Speed",
    "Interpretability",
    "Best For",
    "Advantages",
    "Limitations"
]

random_forest = [
    "Tree Ensemble",
    "Moderate",
    "Fast", 
    "High",
    "Tabular data",
    "Robust, less overfit",
    "Slow w/ huge data"
]

gradient_boosting = [
    "Seq Tree Ens",
    "High",
    "Moderate",
    "Moderate", 
    "High accuracy",
    "Most accurate",
    "Overfits, needs tune"
]

neural_network = [
    "Multi-layer",
    "Very High",
    "Slow",
    "Low",
    "Complex data",
    "Learns complex",
    "Needs more data"
]

# Create the table
fig = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>Characteristic</b>', '<b>Random Forest</b>', '<b>Gradient Boost</b>', '<b>Neural Network</b>'],
        fill_color='#1FB8CD',
        font=dict(color='white', size=14),
        align='center',
        height=40
    ),
    cells=dict(
        values=[characteristics, random_forest, gradient_boosting, neural_network],
        fill_color=[['white']*len(characteristics), 
                   ['#B3E5EC']*len(characteristics),
                   ['#FFCDD2']*len(characteristics), 
                   ['#A5D6A7']*len(characteristics)],
        font=dict(color='black', size=12),
        align='center',
        height=35
    )
)])

fig.update_layout(
    title="ML Algorithm Comparison",
    font=dict(family="Arial", size=12)
)

# Save as both PNG and SVG
fig.write_image("ml_comparison.png")
fig.write_image("ml_comparison.svg", format="svg")