import re
import os
import json
import joblib
from collections import Counter

# Try importing sklearn, but handle cases where it might be missing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =========================================
# SECTION 1: IMPORTING DATA
# =========================================

with open("hallucinations.json", encoding="utf-8") as f:
    hallucinations = [json.loads(line) for line in f if line.strip()]

final_ordered_complexities = []

# =========================================
# SECTION 2: MACHINE LEARNING COMPONENT
# =========================================

# Seed data to bootstrap the model if no external model exists
SEED_PROMPTS = [
    "hi", "hello world", "how are you", "simple text", "the cat sat on the mat",
    "write a story about a dog", "explain 2+2", "what is the weather",
    "analyze the quantum fluctuations of the particle",
    "explain the ontological paradox of existentialism",
    "derive the asymptotic complexity of the recursive algorithm",
    "discuss the geopolitical implications of the industrial revolution",
    "evaluate the thermodynamic entropy in a closed system",
    "critique the chiaroscuro techniques in baroque art",
    "calculate the eigenvector of the matrix using linear algebra"
]

SEED_SCORES = [
    1.0, 1.0, 1.5, 2.0, 2.0,
    3.0, 2.0, 2.5,
    9.0,
    9.5,
    9.0,
    8.5,
    9.0,
    8.5,
    9.0
]

def train_ml_model(training_prompts, training_scores):
    """
    Trains a Linear Regression model using TF-IDF.
    Note: Caller must ensure SKLEARN_AVAILABLE is True.
    """
    print("Training local ML model on seed data...")
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(training_prompts)
    model = LinearRegression()
    model.fit(X, training_scores)
    joblib.dump((vectorizer, model), 'complexity_model.pkl')
    print("Model trained and saved to complexity_model.pkl")


def ml_predict(prompt: str) -> float:
    """
    Predicts complexity using the pre-trained model.
    If the model file is missing, it auto-trains one using SEED_DATA.
    """
    if not SKLEARN_AVAILABLE:
        return 0.0

    model_path = 'complexity_model.pkl'

    # Auto-train if model doesn't exist
    if not os.path.exists(model_path):
        train_ml_model(SEED_PROMPTS, SEED_SCORES)

    try:
        vectorizer, model = joblib.load(model_path)
        X = vectorizer.transform([prompt])
        ml_score = float(model.predict(X)[0])
        return max(0.0, min(10.0, ml_score))
    except Exception as e:
        print(f"ML Prediction Error: {e}")
        return 0.0

# =========================================
# SECTION 3: PROCESS HALLUCINATIONS
# =========================================

def process_hallucinations():
    """
    Processes each hallucination from the loaded data.
    Extracts the 'question' field as the prompt, computes complexity,
    and outputs complexity + justification (list of all sub-scores).
    """
    print(f"Processing {len(hallucinations)} hallucinations...")
    
    for index, entry in enumerate(hallucinations, start=1):
        prompt = entry.get("question", "")
        
        if not prompt.strip():
            final_ordered_complexities.append({
                "complexity": 0,
                "justification": "Empty prompt"
            })
            continue
        
        complexity = ml_predict(prompt)
        
        final_ordered_complexities.append({
            "complexity": f"{complexity:.2f}",
            "justification": f"ML model score: {complexity:.2f}"
            })
        
        if index % 50 == 0 or index == len(hallucinations):
            print(f"   [OK] Processed {index}/{len(hallucinations)}")
    
    # Save results to JSON
    with open("prompt_complexities.json", "w", encoding="utf-8") as f:
        json.dump({"complexities": final_ordered_complexities}, f, indent=2, ensure_ascii=False)
    
    print(f"\nDONE - Complexities saved to prompt_complexities.json")


if __name__ == '__main__':
    process_hallucinations()
