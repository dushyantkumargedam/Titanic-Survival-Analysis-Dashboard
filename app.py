import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

# --- 1. Data Loading and Preprocessing ---
# Load the built-in Titanic dataset
try:
    df = sns.load_dataset('titanic')
except Exception as e:
    # Fallback for deployment environments that might struggle with external dataset loading
    print(f"Error loading Seaborn dataset: {e}. Initializing empty DataFrame.")
    df = pd.DataFrame() 

# --- 2. Data Cleaning and Feature Engineering ---
if not df.empty:
    # Impute missing 'age' with the mean
    mean_age = df['age'].mean()
    df['age'] = df['age'].fillna(mean_age)
    
    # Impute missing 'embarked' with the most common value
    most_common_embarked = df['embarked'].mode()[0]
    df['embarked'] = df['embarked'].fillna(most_common_embarked)
    
    # Feature Engineering
    df['sex_label'] = df['sex'].map({'male': 'Male', 'female': 'Female'})
    
    # Age Group Binning
    bins = [0, 13, 19, 60, 100]
    labels = ['Child', 'Teenager', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False, ordered=True, include_lowest=True)
    
    # Family Size
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    
    # Fare Group Binning 
    try:
        # Create 4 fare quantiles
        fare_bins = pd.qcut(df['fare'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
        df['fare_group'] = fare_bins
    except ValueError:
        # Fallback if fare data is too uniform
        df['fare_group'] = pd.cut(df['fare'], bins=3, labels=['Low', 'Medium', 'High'])


# --- 3. Derived DataFrames ---
# DataFrame for Graph 2: Survivors Only
df_after = df[df['survived'] == 1].copy()


# --- 4. Plotly Helper Functions ---

def get_composition_fig(data_df, feature, title, color_sequence):
    """Generates a bar plot for total counts/composition with fixed height."""
    if data_df.empty:
        return go.Figure().add_annotation(text="No data available for this chart.", showarrow=False)

    fig = px.histogram(
        data_df,
        x=feature,
        color=feature,
        color_discrete_sequence=color_sequence,
        title=f"{title} Distribution (Count)",
        text_auto=True
    )
    # VITAL: Set fixed height in update_layout to match the container style
    fig.update_layout(
        xaxis_title=title,
        yaxis_title="Count of Passengers",
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=40, r=40),
        legend_title_text='Category',
        height=400  # Fixed height for stable layout
    )
    return fig

def get_survival_fig(data_df, feature, title):
    """Generates a bar plot for the survival rate percentage with fixed height."""
    if data_df.empty:
        return go.Figure().add_annotation(text="No data available for this chart.", showarrow=False)

    # Calculate survival rate for the selected feature
    survival_rate_df = df.groupby(feature)['survived'].mean().reset_index()
    survival_rate_df['survival_rate'] = (survival_rate_df['survived'] * 100).round(2)

    # Sort data for better visualization
    survival_rate_df = survival_rate_df.sort_values(by='survival_rate', ascending=False)
    
    fig = px.bar(
        survival_rate_df,
        x=feature,
        y='survival_rate',
        color='survival_rate',
        color_continuous_scale=px.colors.sequential.Sunsetdark,
        title=f"{title} Survival Rate (%)",
        text_auto=True
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    # VITAL: Set fixed height in update_layout to match the container style
    fig.update_layout(
        xaxis_title=title,
        yaxis_title="Survival Rate (%)",
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=40, r=40),
        coloraxis_showscale=False,
        height=500  # Fixed height for stable layout
    )
    return fig

# --- 5. Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# VITAL FOR DEPLOYMENT: Expose the Flask server instance for Gunicorn
server = app.server 

# Dictionary to map dropdown selection to internal column names and display titles
FEATURE_OPTIONS = {
    'Passenger Class': ('pclass', 'Passenger Class'),
    'Sex': ('sex_label', 'Sex'),
    'Age Group': ('age_group', 'Age Group'),
    'Embarked Location': ('embarked', 'Embarked Location'),
    'Family Size': ('family_size', 'Family Size'),
    'Fare Group': ('fare_group', 'Fare Group')
}

# --- 6. Dash Layout ---
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'}, children=[
    html.H1("Titanic Survival Analysis Dashboard", style={'textAlign': 'center', 'color': '#333'}),
    html.Div([
        html.Label("Select Feature for Analysis:", style={'fontWeight': 'bold', 'marginRight': '15px'}),
        dcc.Dropdown(
            id='feature-selector',
            options=[{'label': k, 'value': k} for k in FEATURE_OPTIONS.keys()],
            value='Passenger Class',
            clearable=False,
            style={'width': '300px', 'display': 'inline-block'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div(className='row', style={'display': 'flex', 'flexWrap': 'wrap'}, children=[
        # Graph 1: Total Population
        html.Div(className='six columns', style={'flex': '1 1 300px', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'}, children=[
            html.H4("1. Total Passenger Count", style={'textAlign': 'center', 'color': '#2980b9'}),
            # VITAL: Set fixed height for the dcc.Graph container
            dcc.Graph(id='graph-before', config={'displayModeBar': False}, style={'height': '400px'})
        ]),

        # Graph 2: Survivors Only
        html.Div(className='six columns', style={'flex': '1 1 300px', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'}, children=[
            html.H4("2. Survivors Only Count", style={'textAlign': 'center', 'color': '#27ae60'}),
            # VITAL: Set fixed height for the dcc.Graph container
            dcc.Graph(id='graph-after', config={'displayModeBar': False}, style={'height': '400px'})
        ]),
        
        # Graph 3: Survival Rate (%)
        html.Div(className='twelve columns', style={'flex': '1 1 100%', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'}, children=[
            html.H4("3. Calculated Survival Rate (%)", style={'textAlign': 'center', 'color': '#e74c3c'}),
            # VITAL: Set fixed height for the dcc.Graph container
            dcc.Graph(id='graph-rate', config={'displayModeBar': False}, style={'height': '500px'})
        ])
    ])
])

# --- 7. Dash Callbacks ---

@app.callback(
    Output('graph-before', 'figure'),
    Output('graph-after', 'figure'),
    Output('graph-rate', 'figure'),
    [Input('feature-selector', 'value')]
)
def update_graphs(selected_feature_title):
    """Updates all three graphs based on the dropdown selection."""
    
    # Retrieve the internal column name and display title
    feature_col, display_title = FEATURE_OPTIONS.get(selected_feature_title)
    
    # Graph 1: Before Incident (Total Population)
    fig_before = get_composition_fig(
        data_df=df, 
        feature=feature_col, 
        title=display_title, 
        color_sequence=px.colors.qualitative.D3,
    )

    # Graph 2: After Incident (Survivors Only)
    fig_after = get_composition_fig(
        data_df=df_after, 
        feature=feature_col, 
        title=display_title, 
        color_sequence=px.colors.sequential.Greens_r,
    )

    # Graph 3: Survival Rate (%)
    fig_rate = get_survival_fig(
        data_df=df, 
        feature=feature_col, 
        title=display_title
    )
    
    return fig_before, fig_after, fig_rate
