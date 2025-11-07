import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Included for completeness, though often used for initial loading

## 📊 Titanic Data Visualization Functions ##

# Pie chart to check survived vs not survived
def Survival(data_df):
    """Generates a pie chart for the survival distribution."""
    plt.figure(figsize=(7, 7))
    sizes = data_df['survived'].value_counts()
    labels = ['Did not survive', 'Survived']
    colors = ['#FF6347', '#3CB371'] # Tomato and MediumSeaGreen
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Survival Distribution')
    plt.show()

# Bargraph to show age in titanic passengers
def Age_graph(data_df):
    """Generates a histogram for the age distribution."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data_df['age'].dropna(), bins=30, kde=True, color='#4682B4') # SteelBlue
    plt.title('Age Distribution of Titanic Passengers')
    plt.xlabel('Age')
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for sex in titanic passengers
def sex_graph(data_df):
    """Generates a count plot for the sex distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sex', data=data_df, palette='pastel')
    plt.title('Sex Distribution of Titanic Passengers')
    # Note: Titanic sex column typically has 'male'/'female' strings, not 0/1.
    plt.xlabel('Sex')
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for fare in titanic passengers
def fare_graph(data_df):
    """Generates a histogram for the fare distribution."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data_df['fare'], bins=30, kde=True, color='#FF8C00') # DarkOrange
    plt.title('Fare Distribution of Titanic Passengers')
    plt.xlabel('Fare')
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for pclass in titanic passengers
def pclass_graph(data_df):
    """Generates a count plot for the passenger class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='pclass', data=data_df, palette='muted')
    plt.title('Passenger Class Distribution of Titanic Passengers')
    plt.xlabel('Passenger Class (1=1st, 2=2nd, 3=3rd)')
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for sibling_spouse (sibsp) in titanic passengers
def sibsp_graph(data_df):
    """Generates a count plot for sibling/spouse distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sibsp', data=data_df, palette='bright')
    plt.title('Sibling/Spouse Distribution of Titanic Passengers')
    plt.xlabel('Number of Siblings/Spouses Aboard')
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for parent/child (parch) in titanic passengers
def parch_graph(data_df):
    """Generates a count plot for parent/child distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='parch', data=data_df, palette='dark')
    plt.title('Parent/Child Distribution of Titanic Passengers')
    plt.xlabel('Number of Parents/Children Aboard')
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for embarked in titanic passengers
def embarked_graph(data_df):
    """Generates a count plot for the embarkation port distribution."""
    plt.figure(figsize=(7, 4))
    sns.countplot(x='embarked', data=data_df, palette='Set2')
    plt.title('Embarkation Port Distribution of Titanic Passengers')
    plt.xlabel('Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)')
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for 'who' in titanic passengers
def who_graph(data_df):
    """Generates a count plot for the 'who' (man/woman/child) distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='who', data=data_df, palette='Set1')
    plt.title('Who Distribution of Titanic Passengers')
    plt.xlabel('Who (man, woman, child)')
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for 'alone' in titanic passengers
def alone_graph(data_df):
    """Generates a count plot for the 'alone' distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='alone', data=data_df, palette='Set3')
    plt.title('Alone Distribution of Titanic Passengers')
    # Note: 'alone' is typically boolean (True/False) or int (0/1). 
    # seaborn will handle this for the plot.
    plt.xlabel('Alone (False = No, True = Yes)') 
    plt.ylabel('Number of Passengers')
    plt.show()

# Graph for survival who are alone
def survival_alone_graph(df):
    plt.figure(figsize=(10, 6))
    survival_by_alone = df.groupby('alone')['survived'].mean() * 100
    
    ax = sns.barplot(x=['With Family', 'Alone'], y=survival_by_alone.values)
    plt.title('Survival Rate: Alone vs With Family')
    plt.xlabel('Passenger Status')
    plt.ylabel('Survival Rate (%)')
    
    # Add percentage labels on top of bars
    for i, v in enumerate(survival_by_alone.values):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.show()