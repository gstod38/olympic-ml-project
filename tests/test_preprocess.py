import pytest
import pandas as pd
import numpy as np
from src.preprocess import clean_data, encode_features

@pytest.fixture
def sample_data():
    """Creates a tiny, messy dataframe to test our cleaning logic."""
    data = {
        'Sex': ['M', 'F', 'M'],
        'Age': [25, np.nan, 30],
        'Height': [180, 170, np.nan],
        'Weight': [80, np.nan, 90],
        'Team': ['USA', 'CHN', 'USA'],
        'NOC': ['USA', 'CHN', 'USA'],
        'Year': [2016, 2012, 2016],
        'Season': ['Summer', 'Summer', 'Summer'],
        'City': ['Rio', 'London', 'Rio'],
        'Sport': ['Judo', 'Judo', 'Judo'],
        'region': ['USA', 'China', None],
        'Medal': ['Gold', None, 'Bronze']
    }
    return pd.DataFrame(data)

def test_fill_missing_age(sample_data):
    """Test 1: Verify Age NaNs are filled."""
    cleaned = clean_data(sample_data)
    assert cleaned['Age'].isnull().sum() == 0

def test_medal_encoding(sample_data):
    """Test 2: Verify Medal_Won is created correctly (1 for medal, 0 for None)."""
    cleaned = clean_data(sample_data)
    
    assert cleaned.loc[1, 'Medal_Won'] == 0
    
    assert cleaned.loc[0, 'Medal_Won'] == 1

def test_binary_encoding(sample_data):
    """Test 3: Verify Sex is mapped to numbers."""
    cleaned = clean_data(sample_data)
    encoded = encode_features(cleaned)
    assert encoded['Sex'].dtype in [np.int64, np.float64, int, float]
    assert encoded['Sex'].iloc[0] == 1  

def test_dataframe_shape(sample_data):
    """Test 4: Ensure preprocessing preserves the source columns and adds the target."""
    cleaned = clean_data(sample_data)
    encoded = encode_features(cleaned)
    assert encoded.shape[1] == 13

def test_clean_data_does_not_mutate_original(sample_data):
    """Test 5: Ensure cleaning works on a copy and leaves the original input unchanged."""
    original = sample_data.copy(deep=True)
    _ = clean_data(sample_data)
    pd.testing.assert_frame_equal(sample_data, original)
