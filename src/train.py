from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from src.preprocess import clean_data, encode_features
except ModuleNotFoundError:
    from preprocess import clean_data, encode_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS = [
    "Sex", "Age", "Height", "Weight", "Team", "NOC",
    "Year", "Season", "City", "Sport", "region"
]
MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def load_config():
    """Load hyperparameters and settings from the config file."""
    with open(PROJECT_ROOT / "configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def configure_mlflow(config):
    tracking_path = PROJECT_ROOT / config["mlflow"].get("tracking_path", "mlruns")
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_path.as_uri())


def load_dataset(config):
    """Load, merge, clean, and encode the training dataset."""
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    region_path = PROJECT_ROOT / config["data"]["region_path"]

    events = pd.read_csv(raw_path)
    regions = pd.read_csv(region_path)
    merged = pd.merge(events, regions, on="NOC", how="left")
    cleaned = clean_data(merged)
    encoded = encode_features(cleaned)
    return encoded


def build_model(model_config):
    classifier_name = model_config["classifier"]
    if classifier_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported classifier '{classifier_name}' in config.")
    model_class = MODEL_REGISTRY[classifier_name]
    return model_class(**model_config["params"])


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_scores),
    }


def log_dataset_metadata(config, df):
    mlflow.log_params(
        {
            "data.raw_path": config["data"]["raw_path"],
            "data.region_path": config["data"]["region_path"],
            "data.cleaned_path": config["data"]["cleaned_path"],
            "data.test_size": config["data"]["test_size"],
            "data.random_state": config["data"]["random_state"],
            "data.description": config["data"]["data_version"],
            "preprocessing.scale_numeric_features": config["preprocessing"]["scale_numeric_features"],
            "preprocessing.scaling_rationale": config["preprocessing"]["scaling_rationale"],
            "dataset.row_count": len(df),
            "dataset.feature_count": len(FEATURE_COLUMNS),
        }
    )


def train_models():
    config = load_config()
    configure_mlflow(config)
    print("Loading data...")

    try:
        df = load_dataset(config)
    except FileNotFoundError as e:
        print(f"Error: Required CSV files not found in data/ folder. {e}")
        return []

    X = df[FEATURE_COLUMNS]
    y = df["Medal_Won"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run_summaries = []
    for model_config in config["model_configs"]:
        run_name = model_config["name"]
        print(f"Starting run: {run_name}")

        with mlflow.start_run(run_name=run_name):
            model = build_model(model_config)
            model.fit(X_train, y_train)

            metrics = compute_metrics(model, X_test, y_test)
            mlflow.log_params(
                {
                    "run_name": run_name,
                    "model.classifier": model_config["classifier"],
                    **{f"model.{k}": v for k, v in model_config["params"].items()},
                }
            )
            log_dataset_metadata(config, df)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="olympic_medal_model")

            run_summaries.append((run_name, metrics))

    print("\n==============================")
    print("   TRAINING RUN COMPARISON    ")
    print("==============================")
    for run_name, metrics in run_summaries:
        print(
            f"{run_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )
    print("==============================\n")

    return run_summaries


if __name__ == "__main__":
    train_models()
