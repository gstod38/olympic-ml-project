import os
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# 1. Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

with open(PROJECT_ROOT / "configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

try:
    from src.preprocess import clean_data, encode_features
except ModuleNotFoundError:
    from preprocess import clean_data, encode_features

# 2. API Setup
nebius_api_key = os.environ.get("NEBIUS_API_KEY")
client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=nebius_api_key)
LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# --- NEW: Helper to ensure encoding matches training data ---
def get_training_frequencies():
    """Loads raw data to calculate frequency maps so app.py matches train.py logic."""
    events = pd.read_csv(PROJECT_ROOT / "data/athlete_events.csv")
    regions = pd.read_csv(PROJECT_ROOT / "data/noc_regions.csv")
    df = pd.merge(events, regions, on='NOC', how='left')
    
    freq_maps = {}
    for col in ['Team', 'NOC', 'City', 'Sport', 'region']:
        freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
    return freq_maps

# Calculate these once at startup
print("Initializing encoding maps...")
FREQ_MAPS = get_training_frequencies()

def get_best_model():
    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    if experiment is None:
        raise RuntimeError("MLflow experiment not found. Run `python src/train.py` first.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        raise RuntimeError("No MLflow runs found. Run `python src/train.py` first.")

    best_run = runs.sort_values("metrics.accuracy", ascending=False).iloc[0]
    print(f"Using Model from Run ID: {best_run.run_id}")
    return mlflow.sklearn.load_model(f"runs:/{best_run.run_id}/olympic_medal_model")

model = get_best_model()

FEATURE_ORDER = ['Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Year', 'Season', 'City', 'Sport', 'region']
SPORT_KEYWORDS = {
    "swim", "swimmer", "swimming", "runner", "running", "sprinter", "judo",
    "gymnast", "gymnastics", "boxing", "boxer", "cycling", "cyclist",
    "rowing", "rower", "wrestling", "wrestler", "fencing", "shooting",
    "skiing", "skier", "snowboard", "snowboarding", "hockey", "football",
    "basketball", "volleyball", "tennis", "golf", "archery", "canoe",
    "kayak", "triathlon", "badminton", "handball", "water polo"
}
LOCATION_HINTS = {"usa", "china", "france", "paris", "rio", "tokyo", "london", "beijing"}
OFF_TOPIC_HINTS = {"weather", "temperature", "forecast", "rain", "news", "stock", "price"}

def looks_like_athlete_query(user_query):
    """Quick local guardrail to reject obvious non-athlete prompts before calling the LLM."""
    normalized = user_query.lower()
    tokens = set(re.findall(r"[a-zA-Z]+", normalized))

    if tokens & OFF_TOPIC_HINTS:
        return False

    score = 0
    if re.search(r"\b\d{4}\b", normalized):
        score += 1
    if re.search(r"\b\d+\s*(yo|year old|years old)\b", normalized):
        score += 1
    if {"male", "female", "man", "woman", "boy", "girl"} & tokens:
        score += 1
    if {"athlete", "olympic", "medal"} & tokens:
        score += 1
    if SPORT_KEYWORDS & tokens:
        score += 1
    if LOCATION_HINTS & tokens:
        score += 1

    return score >= 2

def redirect_message():
    return (
        "This app only predicts Olympic medal outcomes from athlete descriptions. "
        "Try something like: '25 year old male swimmer from the USA competing in Rio 2016.'"
    )

def parse_input_with_llm(user_query):
    system_prompt = """
    Return ONLY a JSON object for an Olympic athlete with these keys:
    'Sex' (1 for M, 0 for F), 'Age', 'Height', 'Weight', 'Team', 'NOC', 
    'Year', 'Season' (1 for Summer, 0 for Winter), 'City', 'Sport', 'region'.
    Use the exact strings for Team, NOC, City, and Sport found in the 120 Years of Olympics dataset.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def main():
    print("\n--- Olympic Athlete Medal Predictor ---")
    while True:
        user_input = input("\nDescribe the athlete (or 'quit'): ")
        if user_input.lower() == 'quit': break

        if not looks_like_athlete_query(user_input):
            print(f"\n{redirect_message()}")
            continue
            
        try:
            features_dict = parse_input_with_llm(user_input)
            input_df = pd.DataFrame([features_dict])
            
            # CRITICAL FIX: Manually apply the frequencies from training
            # This ensures 'USA' becomes 0.07, not 1.0
            for col, mapping in FREQ_MAPS.items():
                input_df[col] = input_df[col].map(mapping).fillna(0.0)
            
            # Binary mappings
            input_df['Sex'] = input_df['Sex'].astype(int)
            input_df['Season'] = input_df['Season'].astype(int)
            
            # Ensure column order matches the model's training
            input_df = input_df[FEATURE_ORDER]
            
            prediction = model.predict(input_df)
            
            # Explanation
            result_str = "likely to win a medal" if int(prediction[0]) == 1 else "unlikely to win a medal"
            explanation = client.chat.completions.create(
                model=LLM_MODEL_ID,
                messages=[{"role": "user", "content": f"The model predicts: {result_str}. Profile: {features_dict}. Explain why in 2 sentences."}]
            ).choices[0].message.content
            
            print(f"\nPREDICTION: {'MEDAL' if int(prediction[0]) == 1 else 'NO MEDAL'}")
            print(f"EXPLANATION: {explanation}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
