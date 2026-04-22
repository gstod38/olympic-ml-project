import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

try:
    from src.preprocess import clean_data, encode_features
except ModuleNotFoundError:
    from preprocess import clean_data, encode_features

def load_config():
    """Load hyperparameters and settings from the config file."""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model():
    # 1. Setup & Config
    config = load_config()
    print("Loading data...")
    
    # Merge athlete events with region data
    try:
        events = pd.read_csv("data/athlete_events.csv")
        regions = pd.read_csv("data/noc_regions.csv")
        df = pd.merge(events, regions, on='NOC', how='left')
    except FileNotFoundError as e:
        print(f"Error: Required CSV files not found in data/ folder. {e}")
        return

    # 2. Preprocessing
    print("Cleaning and encoding...")
    df = clean_data(df)
    df = encode_features(df)
    
    # 3. Feature Selection (matches your test suite)
    features = ['Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Year', 'Season', 'City', 'Sport', 'region']
    X = df[features]
    y = df['Medal_Won']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state']
    )
    
    # 4. MLflow Setup
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        params = config['model']
        print(f"Starting run with params: {params}")
        
        # Initialize and train
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # 5. Metrics Calculation
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }
        
        # 6. Logging to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "olympic_medal_model")
        
        print("\n" + "="*30)
        print("      TRAINING SUCCESSFUL     ")
        print("="*30)
        for name, value in metrics.items():
            print(f"{name.capitalize()}: {value:.4f}")
        print("="*30 + "\n")

if __name__ == "__main__":
    train_model()
