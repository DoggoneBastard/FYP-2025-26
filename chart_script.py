import plotly.graph_objects as go
import plotly.express as px

# Create a flowchart using Plotly shapes and annotations
fig = go.Figure()

# Define positions for flowchart elements
positions = {
    'start': (5, 14),
    'clean': (5, 12),
    'split': (5, 10),
    'de': (2, 8),
    'rf': (4, 8),
    'gb': (6, 8),
    'nn': (8, 8),
    'pred': (5, 6),
    'compare': (5, 4),
    'select': (5, 2),
    'lab': (5, 0),
    'decision': (5, -2),
    'use': (8, -4),
    'add': (2, -4),
    'end': (8, -6)
}

# Define colors for different types
colors = {
    'dataprep': '#B3E5EC',
    'modeling': '#A5D6A7', 
    'experiment': '#FFCDD2',
    'decision': '#FFEB8A',
    'endpoint': '#9FA8B0'
}

# Add rectangles for process boxes
boxes = [
    ('start', 'Start: Data in Excel', 'dataprep'),
    ('clean', 'Data Cleaning', 'dataprep'),
    ('split', 'Split Train/Test', 'dataprep'),
    ('de', 'Diff Evolution', 'modeling'),
    ('rf', 'Random Forest', 'modeling'),
    ('gb', 'Gradient Boost', 'modeling'),
    ('nn', 'Neural Network', 'modeling'),
    ('pred', 'Predictions', 'modeling'),
    ('compare', 'Compare Results', 'modeling'),
    ('select', 'Select Top 3-5', 'modeling'),
    ('lab', 'Lab Validation', 'experiment'),
    ('add', 'Add Data', 'experiment'),
    ('use', 'Use Models', 'endpoint'),
    ('end', 'End: Optimal Found', 'endpoint')
]

for box_id, text, box_type in boxes:
    x, y = positions[box_id]
    fig.add_shape(
        type="rect",
        x0=x-0.8, y0=y-0.3, x1=x+0.8, y1=y+0.3,
        fillcolor=colors[box_type],
        line=dict(color="black", width=1)
    )
    fig.add_annotation(
        x=x, y=y, text=text,
        showarrow=False,
        font=dict(size=10)
    )

# Add diamond for decision
x, y = positions['decision']
fig.add_shape(
    type="path",
    path=f"M {x-0.8} {y} L {x} {y+0.4} L {x+0.8} {y} L {x} {y-0.4} Z",
    fillcolor=colors['decision'],
    line=dict(color="black", width=1)
)
fig.add_annotation(
    x=x, y=y, text="Accurate?",
    showarrow=False,
    font=dict(size=10)
)

# Add arrows
arrows = [
    ('start', 'clean'),
    ('clean', 'split'),
    ('split', 'de'),
    ('split', 'rf'),
    ('split', 'gb'),
    ('split', 'nn'),
    ('de', 'pred'),
    ('rf', 'pred'),
    ('gb', 'pred'),
    ('nn', 'pred'),
    ('pred', 'compare'),
    ('compare', 'select'),
    ('select', 'lab'),
    ('lab', 'decision'),
    ('decision', 'use'),
    ('decision', 'add'),
    ('use', 'end')
]

for start, end in arrows:
    x1, y1 = positions[start]
    x2, y2 = positions[end]
    
    fig.add_annotation(
        x=x2, y=y2,
        ax=x1, ay=y1,
        xref="x", yref="y",
        axref="x", ayref="y",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black"
    )

# Add feedback loop arrow
x1, y1 = positions['add']
x2, y2 = positions['clean']
fig.add_annotation(
    x=x2-0.8, y=y2,
    ax=x1, ay=y1,
    xref="x", yref="y",
    axref="x", ayref="y",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="black"
)

# Add YES/NO labels
fig.add_annotation(x=6.5, y=-3, text="YES", showarrow=False, font=dict(size=8))
fig.add_annotation(x=3.5, y=-3, text="NO", showarrow=False, font=dict(size=8))

# Configure layout
fig.update_layout(
    title="ML Workflow for Cryopreservation",
    xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(range=[-7, 15], showgrid=False, zeroline=False, showticklabels=False),
    showlegend=False,
    plot_bgcolor='white'
)

# Save as PNG and SVG
fig.write_image("ml_workflow.png")
fig.write_image("ml_workflow.svg", format="svg")

print("ML workflow flowchart created successfully!")