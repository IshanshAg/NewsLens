# 📰 NewsLens: NLP-Based News Bias & Sentiment Analyzer

An end-to-end NLP system that combines deterministic text extraction, lexicon metrics, deep learning sentiment classification, rule-based linguistic bias detection, and LLM-driven contextual explanations into a Streamlit dashboard.

---

## 🏛 Architecture Overview

Rather than using an LLM as a simple wrapper, **NewsLens** places the LLM at the end of a multi-stage NLP pipeline.

RAW NEWS ARTICLE
                     │
                     ▼
         Text Cleaning & Tokenization (spaCy)
                     │
                     ▼
         Linguistic Feature Extraction
   ┌─────────────────┼─────────────────┐
   ▼                 ▼                 ▼
   TF-IDF Words      Sentiment          Linguistic
(scikit-learn)   (DistilBERT/VADER)  Bias Indicators
      │                 │                 │
      └─────────────────┼─────────────────┘
                        ▼
        Structured JSON Pipeline Output
                        │
                        ▼
            LLM API Engine (GPT-4o-mini)
                        │
                        ▼
                STREAMLIT DASHBOARD


---

## 🛠 System Features

1. **Text Preprocessing & POS/NER Tagging**: Normalizes raw input, performs lemmatization, tracks Part-of-Speech distributions, and identifies Named Entities using `spaCy`.
2. **Dual-Engine Sentiment Analysis**: Combines NLTK `VADER` lexicon analysis with a fine-tuned HuggingFace `DistilBERT` transformer model.
3. **Rule-Based Linguistic Bias Engine**: Detects loaded terminology, absolute quantifiers, and subjective adjectives to compute a normalized **Bias Index Score (0–100)**.
4. **Keyphrase Mining**: Extracts domain-specific key terms using TF-IDF across document sentence boundaries.
5. **LLM Contextual Interpretation**: Feeds structured pipeline JSON into OpenAI's API to generate an objective 3-sentence summary, critical reading guidelines, and framing explanations.

---

## 📂 Project Structure

```text
NewsLens/
│
├── app.py                      # Streamlit User Interface
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
├── .env.example                # Environment Variable Template
├── .gitignore                  # Git Exclusion Rules
│
├── config/
│   └── config.yaml             # Model Hyperparameters & System Config
│
├── data/
│   └── sample_articles/        # Benchmarking & Demo Test Cases
│
├── prompts/
│   └── news_analysis.txt       # Structured Prompt Engineering Template
│
├── src/                        # Core NLP Engine Modules
│   ├── __init__.py
│   ├── preprocessing.py        # Tokenization, Lemmatization, POS, NER, TF-IDF
│   ├── sentiment.py            # VADER & Transformer Pipeline
│   ├── bias_detector.py        # Rule-based Lexicons & Bias Score Index
│   ├── llm_analyzer.py         # OpenAI ChatCompletions Integration
│   └── utils.py                # Environment & YAML Loaders
│
└── tests/                      # Automated Unit Tests
    ├── test_preprocessing.py
    ├── test_sentiment.py
    └── test_llm.py