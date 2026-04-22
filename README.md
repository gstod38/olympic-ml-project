🥇 Olympic Medal Predictor
This application is an end-to-end Machine Learning project that predicts the likelihood of an athlete winning an Olympic medal based on their physical attributes and event details. It features a Random Forest classifier tracked via MLflow and a natural language interface powered by Llama 3.1.

📋 Project Overview
The goal of this project is to provide a conversational way to interact with historical Olympic data. Instead of looking up stats, a user can describe an athlete profile in plain English, and the system will estimate their success probability based on 120 years of historical trends.

🏗 Architecture
The system consists of two integrated layers:

Predictive Engine: A Random Forest model trained on the Olympic Athletes dataset.

LLM Interface: A natural language processing layer using Nebius AI Studio that extracts 11 structured features from user input and explains the model's output conversationally.

🚀 Setup & Installation
1. Prerequisites
Python 3.13+

DVC (Data Version Control)

Nebius AI API Key

2. Installation
Bash
git clone <your-repo-url>
cd olympic-ml-project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Data & API Configuration
Data: Pull the dataset using DVC: dvc pull (or download from Kaggle and place in data/).

Environment: Create a .env file in the root directory:

Plaintext
NEBIUS_API_KEY=your_key_here
🛠 Usage
Training the Model
To run the training pipeline with Experiment Tracking:

Bash
python src/train.py
Running the Application
To launch the interactive LLM interface:

Bash
python src/app.py

Docker
Build and run the app in a container:

Bash
docker build -t olympic-medal-predictor .
docker run --rm -it --env-file .env olympic-medal-predictor

Notes:
- The image includes `data/` and `mlruns/`, so it can use your local dataset and trained model snapshot at build time.
- Rebuild the image after retraining if you want the container to include newer MLflow artifacts.
Running Tests
Bash
pytest tests/ -v
📊 Results Summary
The best-performing model was selected based on its ability to handle class imbalance (as most athletes do not win medals).

Model: Random Forest Classifier

Accuracy: 94.4%

Recall: 88.1% (Crucial for identifying true medalists)

Precision: 77.0%

🧠 Reflection
Challenges
Categorical Alignment: The biggest challenge was ensuring that the "Team" or "Sport" a user typed into the LLM-Powered Interface matched the frequency-encoded values the model was trained on. I solved this by exporting the training frequency maps and applying them dynamically in app.py.

Missing Data: Handling edge cases where users don't provide height or weight required specific prompting strategies to ensure the LLM provided "Olympic average" defaults.

Future Improvements
Implement a Streamlit web dashboard for a better UI.

Incorporate more granular "Event" data to improve precision for specific disciplines.
