import json
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

import spacy
from src.bias_detector import analyze_bias_and_loaded_language
from src.sentiment import analyze_sentiment
from src.utils import load_config

# Load models
config = load_config()
nlp = spacy.load(config['nlp']['spacy_model'])

def run_evaluation(dataset_path="tests/evaluation_dataset.json", bias_threshold=0.02):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    y_true_bias = []
    y_pred_bias = []

    y_true_sentiment = []
    y_pred_sentiment = []

    for item in dataset:
        text = item["text"]
        doc = nlp(text)

        # 1. Bias Detector Prediction with Density Threshold
        bias_res = analyze_bias_and_loaded_language(doc)
        # Check density score threshold instead of simple > 0 check
        pred_has_bias = bias_res["bias_score"] >= bias_threshold
        
        y_true_bias.append(item["ground_truth_has_bias"])
        y_pred_bias.append(pred_has_bias)

        # 2. Sentiment Prediction
        sent_res = analyze_sentiment(text)
        pred_sentiment = sent_res["primary_label"]
        
        y_true_sentiment.append(item["ground_truth_sentiment"])
        y_pred_sentiment.append(pred_sentiment)

    # ---------------------------------------------------------
    # BIAS DETECTOR EVALUATION
    # ---------------------------------------------------------
    print("==========================================")
    print("      BIAS DETECTOR EVALUATION RESULTS    ")
    print("==========================================")
    print(f"Tested Samples: {len(dataset)}")
    print(f"Bias Score Threshold Applied: {bias_threshold}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true_bias, y_pred_bias, target_names=["Unbiased", "Biased"]))
    
    cm_bias = confusion_matrix(y_true_bias, y_pred_bias)
    print("Confusion Matrix:")
    print(f"TN: {cm_bias[0][0]} | FP: {cm_bias[0][1]}")
    print(f"FN: {cm_bias[1][0]} | TP: {cm_bias[1][1]}")

    # ---------------------------------------------------------
    # SENTIMENT MODEL EVALUATION
    # ---------------------------------------------------------
    print("\n==========================================")
    print("      SENTIMENT MODEL EVALUATION RESULTS  ")
    print("==========================================")
    print("\nDetailed Sentiment Classification Report:")
    print(classification_report(y_true_sentiment, y_pred_sentiment, zero_division=0))

if __name__ == "__main__":
    run_evaluation()