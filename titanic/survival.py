import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# --- Graph 1: Survival % by sex (Bar Chart) ---
def get_sex_survival_fig(df):
    fig = px.bar(
        df,
        x='sex_label',
        y='survived',
        text_auto=True,
        color='sex_label',
        color_discrete_map={'Male': '#b3e5fc', 'Female': '#ffb6c1'}, # Pastel-like colors
        title='Survival Percentage by Sex'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Sex', yaxis_title='Survival (%)', showlegend=False)
    return fig

# --- Graph 2: Survival % by age group (Bar Chart) ---
def get_age_survival_fig(df):
    fig = px.bar(
        df,
        x='age_group',
        y='survived',
        text_auto=True,
        color='age_group',
        color_discrete_sequence=px.colors.sequential.Teal, # Muted color-like
        title='Survival Percentage by Age Group'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Age Group', yaxis_title='Survival (%)', showlegend=False)
    return fig

# --- Graph 3: Women survival % by class (Bar Chart) ---
def get_women_class_survival_fig(df):
    fig = px.bar(
        df,
        x='pclass',
        y='survived',
        text_auto=True,
        color='pclass',
        color_discrete_sequence=px.colors.qualitative.Set2,
        title='Women Survival % by Pclass'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Pclass', yaxis_title='Survival (%)', showlegend=False)
    return fig

# --- Graph 4: Survival % by fare group (Bar Chart) ---
def get_fare_survival_fig(df):
    # Use a diverging color scheme like 'coolwarm'
    colors = px.colors.diverging.Coolwarm[:4]
    fig = px.bar(
        df,
        x='fare_group',
        y='survived',
        text_auto=True,
        color='fare_group',
        color_discrete_sequence=colors,
        title='Survival % by Fare Group (quartiles)'
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Fare Group', yaxis_title='Survival (%)', showlegend=False, xaxis={'tickangle': 30})
    return fig

# --- Graph 5: Survival % by family size (Line Plot) ---
def get_family_survival_fig(df):
    fig = px.line(
        df,
        x='family_size',
        y='survived',
        markers=True,
        title='Survival % by Family Size'
    )
    # Add annotations manually for line plot data labels
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row['family_size'],
            y=row['survived'],
            text=f"{row['survived']:.1f}%",
            showarrow=False,
            yshift=10,
            font=dict(size=8)
        )
    fig.update_layout(xaxis_title='Family Size', yaxis_title='Survival (%)')
    return fig

# --- Graph 6: Age distribution (KDE Plot) ---
def get_age_distribution_fig(df):
    # Note: Plotly Express uses 'Violin' for similar density representation or
    # 'Hist' with 'groupnorm'='percent'. A custom go.Figure is best for the KDE style.
    survived_map = {0: 'Not survived (0)', 1: 'Survived (1)'}
    df['survived_label'] = df['survived'].map(survived_map)

    # Use a custom density plot with Plotly Graph Objects
    fig = go.Figure()
    colors = ['#ff9999', '#99ff99']

    # Iterate over survival status
    for i, status in enumerate([0, 1]):
        subset = df[df['survived'] == status]['age'].dropna()
        if not subset.empty:
            fig.add_trace(go.Violin(
                x=subset,
                legendgroup=str(status),
                scalegroup=str(status),
                name=survived_map[status],
                line_color=colors[i],
                fillcolor=colors[i],
                opacity=0.6,
                side='positive',
                bandwidth=20, # Adjust for smoother/sharper KDE
            ))

    fig.update_traces(orientation='h', side='both', width=3, points=False)
    fig.update_layout(
        xaxis_title='Age',
        yaxis_title='Density',
        title='Age Distribution: Survived vs Not Survived',
        violinmode='overlay',
        yaxis={'showticklabels': False, 'title': 'Density'},
        xaxis={'title': 'Age'}
    )
    return fig