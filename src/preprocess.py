import pandas as pd

def clean_data(df):
    # 1. Fill Age, Height, Weight gaps using the medians we calculated
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Height'] = df['Height'].fillna(df.groupby(['Sport', 'Sex'])['Height'].transform('median'))
    df['Height'] = df['Height'].fillna(df['Height'].median())
    df['Weight'] = df['Weight'].fillna(df.groupby(['Sport', 'Sex'])['Weight'].transform('median'))
    df['Weight'] = df['Weight'].fillna(df['Weight'].median())
    
    # 2. Fill missing regions
    df['region'] = df['region'].fillna('Unknown')
    
    # 3. Create Target Variable
    df['Medal_Won'] = df['Medal'].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    return df

def encode_features(df):
    # Binary Encoding
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['Season'] = df['Season'].map({'Summer': 1, 'Winter': 0})
    
    # Frequency Encoding for high-cardinality columns
    categorical_cols = ['Team', 'NOC', 'City', 'Sport', 'region']
    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)
        
    # Drop non-numeric and redundant columns
    cols_to_keep = ['Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 
                    'Year', 'Season', 'City', 'Sport', 'region', 'Medal_Won']
    return df[cols_to_keep]