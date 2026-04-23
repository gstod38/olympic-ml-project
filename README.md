# Olympic Medal Predictor

This project is an end-to-end machine learning application that estimates whether an Olympic athlete profile looks likely to win a medal based on historical Olympic data. It is designed for reviewers who want to see the full ML lifecycle in one repo: preprocessing, model comparison, MLflow tracking, testing, and an LLM-powered interface.

## Project Description

The application solves a simple usability problem: historical Olympic data is hard to query conversationally. Instead of manually assembling model features, a user can describe an athlete in natural language and the system will:

1. extract the relevant features,
2. load the best tracked ML model,
3. return a medal prediction with a short explanation.

The current model should be interpreted as a historical-pattern estimate, not as a production-grade future forecasting tool.

## Architecture Overview

The system has two connected layers:

- `src/train.py` preprocesses the Olympic dataset, trains five model configurations, logs them to MLflow, and stores the trained models as artifacts.
- `src/evaluate.py` uses `mlflow.search_runs()` to rank completed runs by the configured selection metric (`f1`) and evaluate the best run on the held-out test split.
- `src/app.py` provides a CLI interface. It uses Nebius AI Studio to parse athlete descriptions and generate explanations, then passes structured features into the best tracked model.

Preprocessing is defined in `src/preprocess.py`. Missing values are imputed, high-cardinality categoricals are frequency encoded, and numeric features are intentionally left unscaled because the compared models are tree-based and do not require feature scaling.

## Repository Structure

```text
your-project/
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── configs/
│   └── config.yaml
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_interface.py
├── notebooks/
│   └── exploration.ipynb
└── data/
    ├── athlete_events.csv.dvc
    └── noc_regions.csv.dvc
```

## Setup

### Prerequisites

- Python 3.13+
- DVC, or local copies of the Olympic CSV files in `data/`
- A Nebius AI Studio API key if you want the live LLM parsing/explanation path

### Installation

```bash
git clone <your-repo-url>
cd olympic-ml-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data and Environment Configuration

Pull the dataset with DVC or place the CSV files in `data/`:

```bash
dvc pull
```

Create a root `.env` file:

```text
NEBIUS_API_KEY=your_key_here
```

The committed `.env.example` file is only a template. Real secrets should never be committed.

## Usage

### Train and Compare Models

The training script reads five model configurations from `configs/config.yaml`, trains them on the same split, and logs params, dataset metadata, metrics, and model artifacts to MLflow.

```bash
python src/train.py
```

### Evaluate the Best Run

This script ranks MLflow runs by `f1` and evaluates the best model on the held-out test split.

```bash
python src/evaluate.py
```

### Run the Interface

```bash
python src/app.py
```

Example prompt:

```text
25 year old male swimmer from the USA competing in Rio 2016
```

If the input is incomplete or off-topic, the interface redirects the user instead of producing a garbage prediction.

### Run Tests

```bash
pytest tests/ -v
```

Current status:

```text
12 passed
```

## Docker

Build and run the CLI app in a container:

```bash
docker build -t olympic-medal-predictor .
docker run --rm -it --env-file .env olympic-medal-predictor
```

Notes:

- The image copies `src/`, `configs/`, `data/`, and `mlruns/`.
- Rebuild the image after retraining if you want the container to include updated MLflow artifacts.

## Results Summary

Five tracked configurations were compared:

- `rf_baseline`
- `rf_balanced`
- `rf_shallow_balanced`
- `extra_trees_balanced`
- `gradient_boosting`

Best run by `f1`: `rf_balanced`

Held-out test metrics for the selected model:

- Accuracy: `0.8790`
- Precision: `0.5977`
- Recall: `0.5359`
- F1: `0.5651`
- ROC AUC: `0.8688`

Comparison highlights:

- `rf_balanced` produced the strongest overall balance of precision and recall.
- `extra_trees_balanced` delivered similar F1 but slightly lower accuracy.
- `gradient_boosting` kept good precision but collapsed on recall, so it was not selected.

## Reflection

### What I Learned

- MLflow is much more useful when model comparison is driven by a deliberate ranking metric instead of “most recent run”.
- Frequency encoding can work for a fast tabular baseline, but it creates alignment issues between training-time categories and natural-language user inputs.
- A project like this needs guardrails in the interface layer, not just a good model.

### Challenges

- Matching free-form user descriptions to the frequency-encoded categorical features used in training.
- Making tests deterministic even when the live LLM path depends on external API access.
- Cleaning up the MLflow workflow so the best model can be chosen programmatically from multiple runs.

### Future Improvements

- Add athlete-performance features such as rankings, prior medals, and recent results.
- Replace the CLI with a Streamlit or Gradio UI.
- Add a stronger clarification loop for ambiguous prompts instead of a single redirect message.
