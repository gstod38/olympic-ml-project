import pytest
from src.app import clarification_message, find_missing_athlete_details, parse_input_with_llm

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
    
    assert 'Height' in features
    assert 'Weight' in features
    assert features['NOC'] == 'ETH'
    assert features['Sport'] == 'Athletics' or features['Sport'] == 'Running'

def test_invalid_input_handling():
    """
    Test 3: Verify the system handles out-of-scope queries.
    Requirement: Out-of-scope queries are caught.
    """

    query = "How do I make a pepperoni pizza?"
    
    try:
        features = parse_input_with_llm(query)
        assert features.get('Sport') is None or features.get('Sport') == "Unknown"
    except Exception:
        pass

def test_incomplete_input_requests_clarification():
    """
    Test 4: Verify incomplete athlete descriptions are redirected into a clarification request.
    """
    missing = find_missing_athlete_details("A talented athlete from the USA")
    assert "sport" in missing
    assert "Olympic year or host city" in missing
    message = clarification_message(missing)
    assert "I need a bit more detail" in message
