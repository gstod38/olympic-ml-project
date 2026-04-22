import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import the functions we wrote in preprocess.py
from preprocess import clean_data, encode_features

def load_config(config_path="configs/config.yaml"):
    """Loads the settings from our YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    # 1. Load Configuration
    config = load_config()
    
    # 2. Load and Preprocess Data
    print("Loading data...")
    df = pd.read_csv(config['data']['raw_path'])
    
    print("Cleaning and encoding...")
    df = clean_data(df)
    df_ml = encode_features(df)
    
    # 3. Split Features and Target
    X = df_ml.drop(columns=['Medal_Won'])
    y = df_ml['Medal_Won']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'],
        stratify=y
    )

    # 4. MLflow Tracking
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        print(f"Training Random Forest with class_weight={config['model']['class_weight']}...")
        
        # Initialize model with config parameters
        model = RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            class_weight=config['model']['class_weight'],
            n_jobs=config['model']['n_jobs'],
            random_state=config['data']['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 5. Log Everything to MLflow
        mlflow.log_params(config['model'])
        mlflow.log_param("test_size", config['data']['test_size'])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("medal_recall", report['1']['recall'])
        mlflow.log_metric("medal_precision", report['1']['precision'])
        
        # Save the model as an artifact
        mlflow.sklearn.log_model(model, "olympic_medal_model")
        
        print("-" * 30)
        print(f"Model Training Complete!")
        print(f"Accuracy: {acc:.2%}")
        print(f"Medal Recall: {report['1']['recall']:.2%}")
        print("-" * 30)

if __name__ == "__main__":
    train_model()