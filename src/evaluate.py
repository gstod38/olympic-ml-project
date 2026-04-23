from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from src.train import FEATURE_COLUMNS, configure_mlflow, load_dataset
except ModuleNotFoundError:
    from train import FEATURE_COLUMNS, configure_mlflow, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(PROJECT_ROOT / "configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_best_run(runs, selection_metric):
    candidate_metrics = [selection_metric, "recall", "accuracy"]
    for metric_name in candidate_metrics:
        metric_column = f"metrics.{metric_name}"
        if metric_column not in runs.columns:
            continue
        ranked_runs = runs.dropna(subset=[metric_column]).sort_values(metric_column, ascending=False)
        if not ranked_runs.empty:
            best_run = ranked_runs.iloc[0].copy()
            best_run["selected_metric_name"] = metric_name
            return best_run
    raise ValueError("No completed runs contain any supported ranking metric.")


def evaluate_best_model():
    config = load_config()
    configure_mlflow(config)
    experiment_name = config["mlflow"]["experiment_name"]
    selection_metric = config["evaluation"]["selection_metric"]

    print(f"Searching for the best model in experiment: {experiment_name}...")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("Experiment not found. Please run src/train.py first.")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("No runs found in this experiment.")
        return

    best_run = get_best_run(runs, selection_metric)
    best_run_id = best_run["run_id"]
    best_run_name = best_run.get("tags.mlflow.runName", "unnamed_run")

    selected_metric_name = best_run["selected_metric_name"]
    print(
        f"Best run by {selected_metric_name}: {best_run_name} "
        f"({selected_metric_name}={best_run[f'metrics.{selected_metric_name}']:.4f})"
    )
    print(f"Loading Model from Run ID: {best_run_id}")

    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/olympic_medal_model")

    print("Loading data for evaluation...")
    df = load_dataset(config)
    X = df[FEATURE_COLUMNS]
    y = df["Medal_Won"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = predictions

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
    }

    print("\n==============================")
    print("      EVALUATION REPORT       ")
    print("==============================")
    for name, value in metrics.items():
        print(f"{name.upper():<10} {value:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    evaluate_best_model()
