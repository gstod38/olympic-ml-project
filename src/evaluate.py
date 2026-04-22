import mlflow
import pandas as pd

def get_best_model():
    # Search for all runs in your Olympic experiment
    experiment_name = "Olympic_Medal_Prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # Get all runs and sort by 'medal_recall' (our key metric)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    best_run = runs.sort_values("metrics.medal_recall", ascending=False).iloc
    
    print(f"Best Run ID: {best_run['run_id']}")
    print(f"Best Recall: {best_run['metrics.medal_recall']:.4f}")
    
    return best_run['run_id']

if __name__ == "__main__":
    get_best_model()