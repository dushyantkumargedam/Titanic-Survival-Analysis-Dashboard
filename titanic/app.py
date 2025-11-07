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
df = sns.load_dataset('titanic')

# --- Data Cleaning and Imputation ---
mean_age = df['age'].mean()
df['age'] = df['age'].fillna(mean_age)
most_common_embarked = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(most_common_embarked)

# --- Feature Engineering ---
df['sex_label'] = df['sex'].map({'male': 'Male', 'female': 'Female'})
bins = [0, 13, 19, 60, 100]
labels = ['Child', 'Teenager', 'Adult', 'Senior']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False, ordered=True, include_lowest=True)
df['family_size'] = df['sibsp'] + df['parch'] + 1
fare_bins = pd.qcut(df['fare'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
df['fare_group'] = fare_bins

# --- 2. Derived DataFrames ---

# DataFrame for Graph 2: Survivors Only
df_after = df[df['survived'] == 1].copy() 


# --- 3. Helper Functions ---

def calculate_survival_rate(data, group_col):
    """Calculates survival percentage by a given column."""
    grouped_df = data.groupby(group_col, observed=True, as_index=False)['survived'].mean()
    grouped_df['survived'] = grouped_df['survived'] * 100
    return grouped_df

# --- 4. Figure Generation Functions ---

def get_composition_fig(data_df, feature, title, color_sequence, subtitle):
    """
    Creates a count plot for the composition (Graph 1: BEFORE or Graph 2: AFTER).
    Includes the fix for the column naming issue when using reset_index().
    """
    
    # Calculate counts and reset index
    plot_df = data_df[feature].value_counts().reset_index(name='Count')
    
    # Fix for Plotly ValueError: Ensure the column containing categories is named after the feature
    if 'index' in plot_df.columns:
        plot_df = plot_df.rename(columns={'index': feature})
    
    if feature == 'pclass':
        plot_df[feature] = plot_df[feature].astype(str) # Convert pclass to string for categorical axis
        
    fig = px.bar(
        plot_df,
        x=feature,
        y='Count',
        color=feature,
        color_discrete_sequence=color_sequence,
        text_auto=True,
        title=f'{subtitle}: Composition by {title}'
    )
    fig.update_layout(xaxis_title=title, yaxis_title='Total Count', showlegend=False, title_x=0.5)
    return fig

def get_survival_fig(data_df, feature, title):
    """Creates a survival rate plot (Graph 3: RATE) - based on survival.py logic."""
    survival_df = calculate_survival_rate(data_df, feature)

    if feature == 'pclass':
        survival_df[feature] = survival_df[feature].astype(str)
        
    fig = px.bar(
        survival_df,
        x=feature,
        y='survived',
        text_auto=True,
        color=feature,
        color_discrete_sequence=px.colors.sequential.Sunset, # Distinct color for the Rate
        title=f'3. RATE: Survival Rate (%) by {title}'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title=title, yaxis_title='Survival (%)', showlegend=False, title_x=0.5, yaxis_range=[0, 100])
    return fig

# --- 5. Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Dictionary to map dropdown selection to the feature column and its display title
FEATURE_OPTIONS = {
    'Passenger Class': ('pclass', 'Passenger Class'),
    'Sex': ('sex_label', 'Sex'),
    'Age Group': ('age_group', 'Age Group'),
    'Fare Quartile': ('fare_group', 'Fare Quartile'),
    'Family Size': ('family_size', 'Family Size'),
}

# --- 6. Dash Layout (Triple-Pane) ---
app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '20px'}, children=[
    
    html.Div(style={'maxWidth': '1400px', 'margin': '0 auto', 'backgroundColor': 'white', 
                    'padding': '30px', 'borderRadius': '10px', 'boxShadow': '0 4px 12px 0 rgba(0,0,0,0.1)'}, children=[
        
        html.H1(
            children='Titanic Survival Analysis: Population vs. Outcome 🚢',
            style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '20px',
                'borderBottom': '3px solid #3498db',
                'paddingBottom': '10px'
            }
        ),

        # --- Dropdown Control ---
        html.Div([
            html.H3('Select a Key Factor for Analysis:', style={'color': '#34495e', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='feature-selector',
                options=[{'label': k, 'value': k} for k in FEATURE_OPTIONS.keys()],
                value='Passenger Class',  # Default selected value
                clearable=False,
                style={'marginBottom': '30px', 'fontSize': '16px'}
            ),
        ], className='row'),

        # --- Triple-Pane Graph Layout using the 12-column grid ---
        html.Div(className='row', children=[
            # 1. Before Incident (Total Population) -> 4 columns wide
            html.Div(className='four columns', children=[
                html.H4("1. BEFORE: Total Population (df)", style={'textAlign': 'center', 'color': '#2980b9'}),
                dcc.Graph(id='graph-before', config={'displayModeBar': False})
            ]),

            # 2. After Incident (Survivors Only) -> 4 columns wide
            html.Div(className='four columns', children=[
                html.H4("2. AFTER: Survivors Only (df_after)", style={'textAlign': 'center', 'color': '#27ae60'}),
                dcc.Graph(id='graph-after', config={'displayModeBar': False})
            ]),

            # 3. Survival Rate (%) -> 4 columns wide
            html.Div(className='four columns', children=[
                html.H4("3. RATE: Survival Rate (%)", style={'textAlign': 'center', 'color': '#e74c3c'}),
                dcc.Graph(id='graph-rate', config={'displayModeBar': False})
            ])
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
    
    # Graph 1: Before Incident (Total Population) - Blue Palette
    fig_before = get_composition_fig(
        data_df=df, 
        feature=feature_col, 
        title=display_title, 
        color_sequence=px.colors.qualitative.D3, # Blue/Grey colors
        subtitle="1. Total Population"
    )

    # Graph 2: After Incident (Survivors Only) - Green Palette
    fig_after = get_composition_fig(
        data_df=df_after, 
        feature=feature_col, 
        title=display_title, 
        color_sequence=px.colors.sequential.Greens_r, # Green colors
        subtitle="2. Survivors Only"
    )

    # Graph 3: Survival Rate (%) - Red/Orange Palette
    fig_rate = get_survival_fig(
        data_df=df, 
        feature=feature_col, 
        title=display_title
    )
    
    return fig_before, fig_after, fig_rate

# --- 8. Run App ---
if __name__ == '__main__':
    app.run(debug=True)