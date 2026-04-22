import pandas as pd

def clean_data(df):
    """Clean the Olympic dataset by handling missing values."""
    df = df.copy()
    
    # Fill missing values for physical attributes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Height'] = df['Height'].fillna(df['Height'].median())
    df['Weight'] = df['Weight'].fillna(df['Weight'].median())
    
    # Only fill region if the column exists after merging
    if 'region' in df.columns:
        df['region'] = df['region'].fillna('Unknown')
    
    # Create the target variable
    df['Medal_Won'] = df['Medal'].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    return df

def encode_features(df):
    """Apply basic encoding to categorical features."""
    # Convert Sex to binary
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    # Convert Season to binary
    df['Season'] = df['Season'].map({'Summer': 1, 'Winter': 0})
    
    # Use frequency encoding for high-cardinality strings to keep it simple
    cols_to_encode = ['Team', 'NOC', 'City', 'Sport', 'region']
    for col in cols_to_encode:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
            
    return df