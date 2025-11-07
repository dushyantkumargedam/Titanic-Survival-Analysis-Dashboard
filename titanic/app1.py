import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns # Used for easy dataset loading (Titanic dataset)

# --- 1. Data Loading and Preprocessing ---
# This section integrates the data loading, cleaning, and feature engineering 
# steps implied by your titanic.ipynb and required by the survival.py functions.

# Load the built-in Titanic dataset
df = sns.load_dataset('titanic')

# --- Data Cleaning and Imputation ---
# Impute missing age with the mean
mean_age = df['age'].mean()
df['age'] = df['age'].fillna(mean_age)

# Fill missing embarked/embark_town with the most common value ('S' / 'Southampton')
most_common_embarked = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(most_common_embarked)

# --- Feature Engineering ---

# Sex Label
df['sex_label'] = df['sex'].map({'male': 'Male', 'female': 'Female'})

# Age Group (Using common groupings for visualization)
bins = [0, 13, 19, 60, 100]
labels = ['Child', 'Teenager', 'Adult', 'Senior']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False, ordered=True)

# Family Size (sibsp + parch + 1)
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Fare Group (Quartiles)
# 'duplicates="drop"' handles cases where fare values fall on the boundary of bins
fare_bins = pd.qcut(df['fare'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
df['fare_group'] = fare_bins

# --- 2. Aggregate DataFrames for Dash Components ---

def calculate_survival_rate(data, group_col):
    """
    Calculates survival percentage by a given column.
    The fix for the TypeError is included here: only the 'survived' column is multiplied by 100.
    """
    # Group and calculate the mean
    grouped_df = data.groupby(group_col, observed=True, as_index=False)['survived'].mean()
    
    # Multiply ONLY the 'survived' column by 100 to get percentage
    grouped_df['survived'] = grouped_df['survived'] * 100

    return grouped_df

# Pre-calculate data for all charts
sex_df = calculate_survival_rate(df, 'sex_label')
age_df = calculate_survival_rate(df, 'age_group')
pclass_df = calculate_survival_rate(df, 'pclass')
fare_df = calculate_survival_rate(df, 'fare_group')
family_df = calculate_survival_rate(df, 'family_size')

# Separate calculation for women survival by class
women = df[df['sex'] == 'female']
women_class_df = calculate_survival_rate(women, 'pclass')

# --- 3. Figure Generation Functions (Adapted from survival.py) ---
# All functions now return Plotly figure objects, ready for the Dash component.

def get_sex_survival_fig(data_df):
    """Graph 1: Survival % by sex (Bar Chart)"""
    fig = px.bar(
        data_df,
        x='sex_label',
        y='survived',
        text_auto=True,
        color='sex_label',
        color_discrete_map={'Male': '#b3e5fc', 'Female': '#ffb6c1'},
        title='Survival Percentage by Sex'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Sex', yaxis_title='Survival (%)', showlegend=False,
                      title_x=0.5, yaxis_range=[0, 100])
    return fig

def get_age_survival_fig(data_df):
    """Graph 2: Survival % by age group (Bar Chart)"""
    fig = px.bar(
        data_df,
        x='age_group',
        y='survived',
        text_auto=True,
        color='age_group',
        color_discrete_sequence=px.colors.sequential.Teal,
        title='Survival Percentage by Age Group'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Age Group', yaxis_title='Survival (%)', showlegend=False,
                      title_x=0.5, yaxis_range=[0, 100])
    return fig

def get_women_class_survival_fig(data_df):
    """Graph 3: Women survival % by class (Bar Chart)"""
    fig = px.bar(
        data_df,
        x='pclass',
        y='survived',
        text_auto=True,
        color='pclass',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title='Women Survival Percentage by Passenger Class'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Passenger Class', yaxis_title='Survival (%)', showlegend=False,
                      title_x=0.5, yaxis_range=[0, 100])
    return fig

def get_pclass_survival_fig(data_df):
    """Graph 4: Survival % by Passenger Class (Bar Chart)"""
    fig = px.bar(
        data_df,
        x='pclass',
        y='survived',
        text_auto=True,
        color='pclass',
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title='Survival Percentage by Passenger Class'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Passenger Class', yaxis_title='Survival (%)', showlegend=False,
                      title_x=0.5, yaxis_range=[0, 100])
    return fig

def get_fare_survival_fig(data_df):
    """Graph 5: Survival % by Fare Group (Bar Chart)"""
    fig = px.bar(
        data_df,
        x='fare_group',
        y='survived',
        text_auto=True,
        color='fare_group',
        color_discrete_sequence=px.colors.sequential.Sunset,
        title='Survival Percentage by Fare Quartile'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Fare Quartile', yaxis_title='Survival (%)', showlegend=False,
                      title_x=0.5, yaxis_range=[0, 100])
    return fig

def get_family_size_survival_fig(data_df):
    """Graph 6: Survival % by Family Size (Bar Chart)"""
    fig = px.bar(
        data_df,
        x='family_size',
        y='survived',
        text_auto=True,
        color='family_size',
        color_discrete_sequence=px.colors.qualitative.Set1,
        title='Survival Percentage by Family Size'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Family Size', yaxis_title='Survival (%)', showlegend=False,
                      title_x=0.5, yaxis_range=[0, 100])
    return fig

# --- 4. Dash App Setup ---
app = dash.Dash(__name__)

# Dictionary to map dropdown selection to the function and pre-calculated data
GRAPH_OPTIONS = {
    'Survival by Sex': (get_sex_survival_fig, sex_df),
    'Survival by Age Group': (get_age_survival_fig, age_df),
    'Survival by Passenger Class': (get_pclass_survival_fig, pclass_df),
    'Women Survival by Passenger Class': (get_women_class_survival_fig, women_class_df),
    'Survival by Fare Quartile': (get_fare_survival_fig, fare_df),
    'Survival by Family Size': (get_family_size_survival_fig, family_df),
}

# --- 5. Dash Layout (Styling for a cleaner look) ---
app.layout = html.Div(style={'backgroundColor': '#e9eef2', 'minHeight': '100vh', 'padding': '20px'}, children=[
    
    html.Div(style={'maxWidth': '1000px', 'margin': '0 auto', 'backgroundColor': 'white', 
                    'padding': '30px', 'borderRadius': '10px', 'boxShadow': '0 4px 12px 0 rgba(0,0,0,0.1)'}, children=[
        
        html.H1(
            children='Titanic Survival Analysis Dashboard',
            style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '20px',
                'borderBottom': '2px solid #3498db',
                'paddingBottom': '10px'
            }
        ),

        html.Div([
            html.H3('Select a Key Survival Factor:', style={'color': '#34495e', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='graph-selector',
                options=[{'label': k, 'value': k} for k in GRAPH_OPTIONS.keys()],
                value='Survival by Sex',  # Default selected value
                clearable=False,
                style={'marginBottom': '30px', 'fontSize': '16px'}
            ),
        ], style={'padding': '10px'}),
        
        dcc.Graph(id='live-graph', config={'displayModeBar': False}, 
                  style={'border': '1px solid #dcdcdc', 'borderRadius': '5px'})
    ])
])

# --- 6. Dash Callbacks ---
@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-selector', 'value')]
)
def update_graph(selected_graph_title):
    """Updates the dcc.Graph component based on the dropdown selection."""
    
    # Retrieve the function and data associated with the selected title
    fig_func, data_to_plot = GRAPH_OPTIONS.get(selected_graph_title)
    
    # Generate the Plotly figure
    return fig_func(data_to_plot)

# --- 7. Run App ---
if __name__ == '__main__':
    # CORRECTED: Using app.run() for modern Dash versions (fix for previous error)
    app.run(debug=True)