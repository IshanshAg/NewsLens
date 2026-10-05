## **NewsLens — NLP-Based News Bias & Sentiment Analyzer**

An end-to-end Natural Language Processing system for analyzing sentiment polarity, linguistic loadedness, and framing in news media, augmented with automated LLM synthesis via Google's Gemini API.


## Author & Academic Information

- **Name:** Ishansh Agarwal
- **Registration Number:** 23FE10CDS00362
- **Branch:** Data Science
- **Batch:** Batch E
- **Project Title:** NewsLens — NLP-Based News Bias & Sentiment Analyzer
- **GitHub Username:** @IshanshAg (https://github.com/IshanshAg)
- **Training Program:** DATA SCIENCE
- **Institution:** Manipal University Jaipur


## **Executive Summary & Objectives**

NewsLens provides an automated, objective NLP diagnostic pipeline for media literacy. Rather than arbitrarily declaring articles biased, NewsLens identifies observable linguistic indicators—such as sensational terminology, subjective qualifiers, absolute statements, and emotional tone—and synthesizes them using a hybrid NLP architecture.

**Key Objectives**

1. Deterministic Linguistic Profiling: Extract POS counts, named entities, and TF-IDF keyphrases using spaCy and scikit-learn.
2. Dual-Engine Sentiment Analysis: Cross-validate lexicon-based polarity (VADER) against deep-learning contextual models (DistilBERT).
3. 4-Category Lexicon Bias Engine: Detect loaded, sensational, subjective, and absolute terms to compute a normalized density score.
4. Contextual LLM Interpretation: Prompt Gemini 2.5 Flash with rule-based NLP evidence to generate neutral framing summaries and media literacy guidance.


## **Architecture & Pipeline Overview**
- ┌───────────────────────────┐
- │     Input News Article    │
- └─────────────┬─────────────┘
-               │
-               ▼
- ┌───────────────────────────┐
- │   spaCy Preprocessing     │
- │   & Linguistic Layer      │
- └─────────────┬─────────────┘
-               │
-     ┌─────────┼─────────┐
-     │         │         │
-     ▼         ▼         ▼
- ┌───────┐ ┌───────┐ ┌───────┐
- │Senti- │ │ Bias  │ │TF-IDF │
- │ ment  │ │Engine │ │Keys   │
- └───┬───┘ └───┬───┘ └───┬───┘
-     │         │         │
-     └─────────┼─────────┘
-               │
-               ▼
- ┌───────────────────────────┐
- │ Structured Pipeline Data  │
- └─────────────┬─────────────┘
-               │
-               ▼
- ┌───────────────────────────┐
- │  Google Gemini 2.5 LLM    │
- └─────────────┬─────────────┘
-               │
-               ▼
- ┌───────────────────────────┐
- │    Streamlit Dashboard    │
- └───────────────────────────┘


## **Quantitative Model Evaluation**

NewsLens was evaluated against a benchmark dataset (tests/evaluation_dataset.json) containing 20 real-world news snippets covering politics, economics, disaster reporting, and neutral announcements.

1. **Rule-Based Bias Detector Performance (Threshold: 0.02)**

- **Class**	**Precision**	**Recall**	**F1-Score**	**Support**
- Unbiased	0.91	0.91	0.91	11
- Biased	0.89	0.89	0.89	9
- Accuracy	—	—	0.90	20
- Macro Average	0.90	0.90	0.90	20

Confusion Matrix: 
- True Negatives: 10 
- False Positives: 1 
- False Negatives: 1 
- True Positives: 8

2. **Dual Sentiment Model Performance (Overall Accuracy: 85.0% | Weighted F1: 0.85)**

- **Class**	**Precision**	**Recall**	**F1-Score**	**Support**
- NEGATIVE	1.00	0.82	0.90	11
- NEUTRAL	0.71	1.00	0.83	5
- POSITIVE	0.75	0.75	0.75	4

3. **LLM Qualitative Evaluation Rubric (N=10 Runs)**

- **Evaluation Dimension**	**Average Score (1–5)**	**Diagnostic Notes**
- Relevance	4.8 / 5.0	Accurately identifies prioritized narrative angles.
- Grounding	4.7 / 5.0	Strictly anchors claims to NLP metrics provided in prompt context.
- Neutrality	4.9 / 5.0	Maintains analytical, non-judgmental diagnostic tone.
- Hallucination Rate	4.8 / 5.0	Zero fabricated quotes or false terminology observed.
- Completeness	4.6 / 5.0	Fully covers framing, stance, and critical guidance.
- Readability & Structure	4.9 / 5.0	Strict markdown section adherence without unnecessary conversational filler.


## **Repository Structure**

- NewsLens/
- │
- ├── app.py                     # Streamlit Interactive Web Application
- ├── evaluate_pipeline.py       # Automated scikit-learn Evaluation Script
- ├── requirements.txt           # Project Dependencies
- ├── evaluation.txt             # Generated Benchmark Evaluation Metrics Report
- ├── .env.example               # Safe Template for Environment Variables
- │
- ├── config/
- │   └── config.yaml            # Model & Pipeline Configurations
- │
- ├── data/
- │   └── sample_articles/       # Preset News Benchmark Texts
- │
- ├── prompts/
- │   └── news_analysis.txt      # Gemini LLM Prompt Template
- │
- ├── src/
- │   ├── __init__.py
- │   ├── preprocessing.py       # spaCy Tokenization, POS, NER, and TF-IDF
- │   ├── sentiment.py           # VADER & DistilBERT Sentiment Classifiers
- │   ├── bias_detector.py       # 4-Category Lexicon Bias & Density Engine
- │   ├── llm_explainer.py       # Google Gemini API Prompt Engine
- │   └── utils.py               # YAML Config & Helper Functions
- │
- └── tests/
-     └── evaluation_dataset.json # Ground-Truth Evaluation Benchmark (N=20)


## **Quickstart & Installation**

1. Clone the Repository

git clone https://github.com/IshanshAg/NewsLens.git
cd NewsLens

2. Set Up Virtual Environment

Windows:
python -m venv venv
.\venv\Scripts\Activate.ps1

Linux / macOS:
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies & Download spaCy Model

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

4. Configure API Keys

Create a .env file in the root directory:
GEMINI_API_KEY="your_google_gemini_api_key_here"

5. Launch Application

streamlit run app.py
Access at http://localhost:8501


## **Limitations & Future Scope**

**Limitations**
- Rule-Based Lexicon Boundary: Context-dependent irony, sarcasm, or subtle implicit framing can bypass keyword-matching rules.
- Visual Media Exclusion: Analyzes text content only; does not assess headline image framing or video editing.

**Future Scope**
- Supervised Bias Fine-Tuning: Fine-tune a domain-specific BERT variant (e.g., MediaBERT) on annotated news bias corpora.
- Multi-Source Cross-Comparison: Support side-by-side linguistic comparison of two articles covering the same news event.
- Multilingual Support: Extend spaCy and LLM evaluation to non-English media outlets.


## **License & Citation**

This project is developed for academic and portfolio demonstration purposes under Data Science coursework at Manipal University Jaipur.
