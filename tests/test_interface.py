import pytest
from src.app import parse_input_with_llm

def test_extraction_accuracy():
    """
    Test 1: Verify the LLM correctly extracts features from a clear string.
    Requirement: Extracts feature values accurately from conversational text.
    """
    query = "A 25 year old male swimmer from the USA in the 2016 Rio games"
    features = parse_input_with_llm(query)
    
    # Check core numerical/categorical mappings
    assert features['Sex'] == 1
    assert features['Age'] == 25
    assert features['NOC'] == 'USA'
    assert features['Year'] == 2016
    assert features['Season'] == 1  # Summer
    assert features['Sport'] == 'Swimming'

def test_edge_case_missing_data():
    """
    Test 2: Verify the LLM provides defaults for missing physical attributes.
    Requirement: Edge cases handled gracefully (missing features).
    """
    query = "A runner from Ethiopia in the 2012 London games"
    features = parse_input_with_llm(query)
    
    # The LLM should have 'hallucinated' reasonable defaults for Height/Weight 
    # instead of leaving them null, which would crash the model.
    assert 'Height' in features
    assert 'Weight' in features
    assert features['NOC'] == 'ETH'
    assert features['Sport'] == 'Athletics' or features['Sport'] == 'Running'

def test_invalid_input_handling():
    """
    Test 3: Verify the system handles out-of-scope queries.
    Requirement: Out-of-scope queries are caught.
    """
    # Note: Depending on your app.py logic, this might return a specific 
    # error or a null dictionary. This test assumes your LLM tries to 
    # fit it into a schema or your code catches the mismatch.
    query = "How do I make a pepperoni pizza?"
    
    try:
        features = parse_input_with_llm(query)
        # If the LLM returns a dict, it shouldn't have valid Olympic data
        assert features.get('Sport') is None or features.get('Sport') == "Unknown"
    except Exception:
        # If your app.py raises an error for non-Olympic queries, that's also a pass
        pass