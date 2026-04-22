import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score

try:
    from src.preprocess import clean_data, encode_features
except ModuleNotFoundError:
    from preprocess import clean_data, encode_features

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate_best_model():
    config = load_config()
    experiment_name = config['mlflow']['experiment_name']
    
    print(f"Searching for the best model in experiment: {experiment_name}...")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("Experiment not found. Please run src/train.py first.")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print("No runs found in this experiment.")
        return

    # Use the most recent run in the experiment.
    best_run = runs.sort_values("start_time", ascending=False).iloc[0]
    best_run_id = best_run.run_id
    
    print(f"Loading Model from Run ID: {best_run_id}")
    
    # Load the model artifact
    model_uri = f"runs:/{best_run_id}/olympic_medal_model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Load and prepare data
    print("Loading data for evaluation...")
    events = pd.read_csv("data/athlete_events.csv")
    regions = pd.read_csv("data/noc_regions.csv")
    df = pd.merge(events, regions, on='NOC', how='left')
    
    df = clean_data(df)
    df = encode_features(df)
    
    features = ['Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Year', 'Season', 'City', 'Sport', 'region']
    X = df[features]
    y = df['Medal_Won']
    
    # Generate fresh metrics
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    prec = precision_score(y, predictions)
    rec = recall_score(y, predictions)
    
    print("\n==============================")
    print("      EVALUATION REPORT       ")
    print("==============================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("==============================\n")

if __name__ == "__main__":
    evaluate_best_model()
