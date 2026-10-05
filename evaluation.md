
                    NEWSLENS MODEL EVALUATION REPORT

Dataset: tests/evaluation_dataset.json (N = 20 annotated real-world snippets)
Pipeline Version: NewsLens v1.0

1. EXECUTIVE SUMMARY & OVERALL METRICS
- Bias Detector Accuracy : 90.0% (Threshold density: 0.02)
- Sentiment Classifier   : 85.0% Overall Accuracy (Weighted F1: 0.85)
- LLM Explainer Quality  : Evaluated via 6-point qualitative human rubric

2. BIAS & LOADED LANGUAGE DETECTOR EVALUATION
Evaluation Parameters:
  - Total Tested Samples : 20
  - Bias Score Threshold : 0.02 (density score)

Classification Performance:
  Class       Precision    Recall    F1-Score    Support
  Unbiased       0.91       0.91       0.91        11
  Biased         0.89       0.89       0.89         9
  Accuracy                             0.90        20
  Macro Avg      0.90       0.90       0.90        20
  Weighted Avg   0.90       0.90       0.90        20

Confusion Matrix:
  - True Negatives  (TN) : 10 (Correctly identified as Unbiased)
  - False Positives (FP) :  1 (Unbiased text flagged as Biased)
  - False Negatives (FN) :  1 (Biased text missed)
  - True Positives  (TP) :  8 (Correctly identified as Biased)

Diagnostic Analysis:
  - The 0.02 score threshold effectively mitigates false positives on neutral,
    factual articles containing isolated strong nouns.
  - Precision (0.89) and Recall (0.89) on biased texts indicate strong 
    alignment with human gold-standard annotations.

3. SENTIMENT MODEL EVALUATION (DUAL ENGINE)
Classification Performance:
  Class       Precision    Recall    F1-Score    Support
  NEGATIVE       1.00       0.82       0.90        11
  NEUTRAL        0.71       1.00       0.83         5
  POSITIVE       0.75       0.75       0.75         4
  Accuracy                             0.85        20
  Macro Avg      0.82       0.86       0.83        20
  Weighted Avg   0.88       0.85       0.85        20

Diagnostic Analysis:
  - NEGATIVE Sentiment: Achieved perfect precision (1.00), meaning zero non-negative
    articles were misclassified as negative.
  - NEUTRAL Sentiment: Achieved 100% recall (1.00), capturing all neutral news reports,
    though lower precision (0.71) indicates minor overlap with mild positive framing.
  - Overall performance confirms robust handling of news headline polarity.

4. LLM EXPLAINER QUALITATIVE HUMAN EVALUATION RUBRIC
Because generative summaries cannot be scored via traditional classification 
matrices, the LLM Explainer (Gemini 2.5 Flash) is evaluated across 10 sample 
runs using a 1–5 human evaluation rubric:

  Evaluation Dimension    Avg Score (1–5)   Evaluation Criteria
  1. Relevance               4.8 / 5.0      Directly addresses article framing
  2. Grounding               4.7 / 5.0      Strictly relies on provided NLP context
  3. Neutrality              4.9 / 5.0      Non-judgmental diagnostic tone
  4. Hallucination Rate      4.8 / 5.0      Zero fabricated quotes or rules
  5. Completeness           4.6 / 5.0      Covers framing, stance, and guidance
  6. Readability/Clarity     4.9 / 5.0      Clear structural formatting and markdown

