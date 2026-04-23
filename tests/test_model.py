import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import recall_score


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import clean_data, encode_features
from src.evaluate import get_best_run
from src.train import configure_mlflow

mlflow = pytest.importorskip("mlflow")
import mlflow.sklearn

TRAINING_FEATURES = [
    'Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC',
    'Year', 'Season', 'City', 'Sport', 'region'
]

def test_model_prediction_output():
    """Test 5: Verify the model produces binary outputs (0 or 1)."""
    # Load experiment name from same config training uses
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    configure_mlflow(config)

    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    if experiment is None:
        pytest.skip("No MLflow experiment found. Run 'python src/train.py' first.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        pytest.skip("No trained model found. Run 'python src/train.py' first.")

    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/olympic_medal_model"
    model = mlflow.sklearn.load_model(model_uri)

    dummy_input = pd.DataFrame([{
        'Sex': 1, 'Age': 25, 'Height': 180, 'Weight': 80,
        'Team': 0.01, 'NOC': 0.01, 'Year': 2016, 'Season': 1,
        'City': 0.01, 'Sport': 0.01, 'region': 0.01
    }])

    prediction = model.predict(dummy_input)

    # Verify output is a single binary prediction.
    assert len(prediction) == 1
    assert int(prediction[0]) in [0, 1]

def test_model_feature_consistency():
    """Test 6: Verify the model handles exactly 11 features."""
    expected_features = 11
    df_dummy = pd.DataFrame(np.random.randint(0,10,size=(5, 16)), columns=[
        'ID', 'Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 
        'Games', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal', 'region'
    ])
    df_dummy['Sex'] = 'M'
    df_dummy['Season'] = 'Summer'
    
    cleaned = clean_data(df_dummy)
    encoded = encode_features(cleaned)
    features = encoded[TRAINING_FEATURES]
    
    assert features.shape[1] == expected_features

def test_model_recall_threshold():
    """Test 7: Verify the saved best model clears a minimum recall threshold."""
    import yaml
    from sklearn.model_selection import train_test_split

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    configure_mlflow(config)

    experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if experiment is None:
        pytest.skip("No MLflow experiment found. Run 'python src/train.py' first.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    try:
        best_run = get_best_run(runs, config["evaluation"]["selection_metric"])
    except ValueError:
        pytest.skip("No ranked MLflow runs found. Run 'python src/train.py' first.")

    best_run_id = best_run["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/olympic_medal_model")

    events = pd.read_csv("data/athlete_events.csv")
    regions = pd.read_csv("data/noc_regions.csv")
    df = pd.merge(events, regions, on="NOC", how="left")
    df = clean_data(df)
    df = encode_features(df)

    X = df[TRAINING_FEATURES]
    y = df["Medal_Won"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    predictions = model.predict(X_test)
    assert recall_score(y_test, predictions, zero_division=0) >= 0.50
