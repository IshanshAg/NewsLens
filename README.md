**NewsLens — NLP-Based News Bias & Sentiment Analyzer**
An end-to-end Natural Language Processing system for analyzing the sentiment, linguistic patterns, and potential bias indicators in news articles, with contextual interpretation powered by an LLM.

## Student Information
 
**Name** - Ishansh Agarwal 
**Registration Number** - 23FE10CDS00362 
**Branch** - Data Science 
**Batch** - E 
**Project Title** - NewsLens — NLP-Based News Bias & Sentiment Analyzer 
**GitHub Username** - [@IshanshAg](https://github.com/IshanshAg) 
**Training Program** - DATA SCIENCE 

# 1. Project Overview

NewsLens is an NLP-based news analysis application that analyzes a news article from multiple linguistic perspectives.

The system combines traditional Natural Language Processing techniques, machine-learning-based sentiment analysis, rule-based linguistic analysis, TF-IDF feature extraction, and Large Language Model (LLM) interpretation.

The objective is **not to declare an article objectively biased or unbiased**. Instead, NewsLens identifies observable linguistic indicators that may influence how information is presented and provides contextual explanations to help users critically evaluate the article.

The application is implemented as an interactive Streamlit dashboard.

# 2. Problem Statement

News articles can contain emotionally charged language, subjective descriptions, exaggerated claims, and framing choices that may influence how readers perceive an event.

Manual identification of these linguistic patterns can be time-consuming and subjective.

NewsLens addresses this problem by providing an automated NLP pipeline that:

- preprocesses news article text;
- identifies linguistic characteristics;
- analyzes sentiment;
- detects potential linguistic bias indicators;
- extracts important keyphrases;
- identifies named entities;
- calculates a normalized bias indicator score;
- and uses an LLM to provide contextual explanations and a neutral summary.


# 3. Objectives

The main objectives of NewsLens are:

1. To preprocess and normalize news article text using NLP techniques.
2. To perform sentiment analysis using both lexicon-based and transformer-based approaches.
3. To identify potentially loaded, subjective, or absolute language.
4. To extract important terms and keyphrases using TF-IDF.
5. To identify named entities and linguistic patterns using spaCy.
6. To calculate a normalized linguistic bias indicator score.
7. To integrate an LLM through an API for contextual interpretation.
8. To generate a concise and neutral article summary.
9. To provide critical-reading recommendations based on detected linguistic patterns.
10. To provide an interactive interface through Streamlit.


# 🏗️ 4. System Architecture

text
                    ┌─────────────────────┐
                    │    News Article     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text Preprocessing  │
                    │       spaCy         │
                    └──────────┬──────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Linguistic Feature Layer  │
                ├──────────────────────────────┤
                │ • Tokenization              │
                │ • Lemmatization              │
                │ • POS Tagging                │
                │ • Named Entity Recognition   │
                │ • TF-IDF Keyphrases          │
                └──────────────┬───────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌────────────┐   ┌──────────────┐   ┌──────────────┐
      │ Sentiment  │   │ Bias         │   │ TF-IDF       │
      │ Analysis   │   │ Detection    │   │ Keyphrases   │
      ├────────────┤   ├──────────────┤   ├──────────────┤
      │ VADER      │   │ Loaded words │   │ Important    │
      │ DistilBERT │   │ Subjective   │   │ terms        │
      └──────┬─────┘   │ language     │   └──────┬───────┘
             │          │ Absolute     │          │
             │          │ expressions  │          │
             │          └──────┬───────┘          │
             └─────────────────┼──────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Structured NLP     │
                    │ Pipeline Output    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     LLM API         │
                    │ Contextual Analysis │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘


# 5. NLP Methodology

## 5.1 Text Preprocessing

The preprocessing pipeline uses spaCy to prepare the article for downstream analysis.

The pipeline includes:

- text normalization;
- tokenization;
- lemmatization;
- Part-of-Speech (POS) analysis;
- Named Entity Recognition (NER).

The processed text is then passed to the different analytical modules.

## 5.2 Sentiment Analysis

NewsLens uses a dual sentiment-analysis approach.

### VADER

VADER is a lexicon- and rule-based sentiment analysis system designed for textual sentiment analysis.

It provides sentiment-related scores based on the occurrence and intensity of sentiment-bearing terms.

### DistilBERT

NewsLens also uses a transformer-based DistilBERT sentiment model.

This provides a deep-learning-based sentiment prediction and allows comparison with the lexicon-based VADER result.

Using two approaches provides a useful cross-check between traditional NLP and transformer-based analysis.

# 6. Linguistic Bias Detection

NewsLens uses a rule-based linguistic bias engine.

The system looks for observable language patterns including:

- loaded terminology;
- subjective adjectives;
- absolute quantifiers;
- strongly evaluative language;
- other predefined linguistic indicators.

The detected indicators are aggregated into a normalized **Bias Index Score from 0 to 100**.

### Important interpretation

The Bias Index is an **indicator of potentially biased linguistic patterns**, not a definitive measurement of whether an article is factually or politically biased.

A high score means that the text contains more of the predefined linguistic indicators. It does not automatically mean that the article is false, politically biased, or unreliable.


# 7. Keyphrase Extraction

NewsLens uses TF-IDF to identify important terms and keyphrases within the article.

TF-IDF considers:

- how frequently a term occurs in the document;
- how distinctive that term is relative to the document representation.

This helps identify important article-specific terminology.


# 8. Named Entity Recognition

spaCy's Named Entity Recognition is used to identify entities such as:

- people;
- organizations;
- locations;
- dates;
- and other supported entity categories.

This provides additional context about the subjects discussed in the article.


# 9. LLM Integration

The LLM is not used as a replacement for the complete NLP pipeline.

Instead, NewsLens first performs deterministic and model-based NLP analysis and then passes the structured results to the LLM.

The LLM receives information such as:

- article text;
- sentiment results;
- detected linguistic indicators;
- extracted keyphrases;
- named entities;
- bias-related observations.

The LLM then provides:

1. A concise neutral summary.
2. An explanation of sentiment.
3. Interpretation of potential bias indicators.
4. Framing analysis.
5. Critical-reading guidelines.

This architecture makes the LLM a **contextual interpretation layer** rather than a simple chatbot wrapper.


# 10. Prompt Engineering

The main prompt is stored separately in:

text
prompts/news_analysis.txt

The prompt is designed to:

- define the LLM's role;
- provide clear analysis instructions;
- constrain the model to the supplied evidence;
- reduce hallucination;
- distinguish sentiment from bias;
- prevent unsupported political or ideological conclusions;
- produce structured output.

Keeping the prompt outside the Python source code makes it easier to modify and evaluate independently.

# 11. Application Interface

NewsLens uses Streamlit to provide an interactive web interface.

The user can:

1. Enter a news article.
2. Run the NLP analysis.
3. View sentiment results.
4. View detected linguistic bias indicators.
5. View extracted entities and keyphrases.
6. View the calculated bias indicator score.
7. View the LLM-generated contextual analysis.

# 12. Project Structure

text
NewsLens/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── config/
│   └── config.yaml
│
├── data/
│   └── sample_articles/
│
├── prompts/
│   └── news_analysis.txt
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── bias_detector.py
│   ├── llm_analyzer.py
│   └── utils.py
│
└── tests/
    ├── test_preprocessing.py
    ├── test_sentiment.py
    └── test_llm.py


# 13. Technologies Used

| Technology | Purpose |
| Python | Core programming language |
| Streamlit | Web application interface |
| spaCy | NLP preprocessing, POS and NER |
| NLTK | VADER sentiment analysis |
| Transformers | DistilBERT sentiment analysis |
| PyTorch | Transformer model backend |
| scikit-learn | TF-IDF and numerical NLP features |
| Pandas | Data handling |
| NumPy | Numerical operations |
| PyYAML | Configuration management |
| python-dotenv | Environment variable management |
| OpenAI API | LLM-based contextual analysis |

# 14. Installation Guide

## Prerequisites

Before installing NewsLens, make sure you have:

- Python 3.10 or compatible Python version
- Git
- Internet connection
- An OpenAI API key for LLM functionality

## Step 1 — Clone the repository

bash
git clone https://github.com/IshanshAg/NewsLens.git
cd NewsLens

## Step 2 — Create a virtual environment

### Windows

powershell
python -m venv venv


Activate it:

powershell
.\venv\Scripts\Activate.ps1

If PowerShell blocks script execution:

powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned


Then activate again:

powershell
.\venv\Scripts\Activate.ps1


You should see:

text
(venv)


at the beginning of your terminal.

## Step 3 — Install dependencies

bash
pip install --upgrade pip
pip install -r requirements.txt


## Step 4 — Download the spaCy language model

Run:

bash
python -m spacy download en_core_web_sm


This model is required for the spaCy NLP pipeline.



## Step 5 — Configure the API key

Create a file named:

text
.env


in the root project directory.

Add:

text
OPENAI_API_KEY=your_api_key_here

Replace the placeholder with your actual API key.

### Security

Never commit `.env` to GitHub.

The repository contains `.env.example` as a safe template.


## Step 6 — Verify the project structure

Your directory should contain:

text
NewsLens/
├── app.py
├── requirements.txt
├── .env
├── .env.example
├── config/
├── data/
├── prompts/
├── src/
└── tests/

## Step 7 — Run the application

Start Streamlit:

bash
streamlit run app.py

The application will normally be available at:

text
http://localhost:8501



# 15. How to Use NewsLens

1. Start the Streamlit application.
2. Enter or paste a news article into the input area.
3. Start the analysis.
4. Review the preprocessing and linguistic information.
5. Review the sentiment analysis results.
6. Review detected potential bias indicators.
7. Review the bias indicator score.
8. Examine extracted keyphrases and named entities.
9. Read the LLM-generated contextual explanation.
10. Use the critical-reading notes to interpret the article more carefully.

# 16. Results & Evaluation

Evaluation is performed at multiple levels because NewsLens contains several different components.

## 16.1 Sentiment Evaluation

The sentiment component should be evaluated using a labelled test set containing positive, negative, and/or neutral examples as supported by the implemented model.

Recommended metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Results

| Metric | VADER | DistilBERT |
| Accuracy | `TO BE MEASURED` | `TO BE MEASURED` |
| Precision | `TO BE MEASURED` | `TO BE MEASURED` |
| Recall | `TO BE MEASURED` | `TO BE MEASURED` |
| F1-score | `TO BE MEASURED` | `TO BE MEASURED` |

> **Important:** These values should be replaced with measurements obtained by running the evaluation code. No performance values should be manually estimated.


## 16.2 Bias Detection Evaluation

The bias detector is rule-based and therefore should not be evaluated in exactly the same way as a supervised classifier unless a labelled dataset is available.

For a manually annotated test set, evaluate:

- True Positives
- False Positives
- True Negatives
- False Negatives
- Precision
- Recall
- F1-score

Example evaluation table:

| Metric | Result |
| Test Articles | `N` |
| Detected Indicators | `N` |
| Correctly Identified Indicators | `N` |
| Precision | `N%` |
| Recall | `N%` |
| F1-score | `N%` |

## 16.3 LLM Evaluation

Because the LLM produces generated explanations rather than fixed classification labels, evaluation should focus on output quality.

The following criteria can be manually assessed:

| Criterion | Evaluation |
| Relevance | Does the explanation address the article? |
| Grounding | Are claims supported by the supplied text? |
| Clarity | Is the explanation understandable? |
| Neutrality | Does it avoid unsupported political conclusions? |
| Hallucination | Does it introduce information absent from the article? |
| Completeness | Does it address the requested analysis sections? |

A small manually reviewed test set can be used to calculate the percentage of outputs satisfying each criterion.

Example:
text
LLM Evaluation Set: N articles

Grounded responses: N/N
Relevant responses: N/N
Neutral responses: N/N
Hallucination-free responses: N/N
Complete responses: N/N


Again, replace `N` with your actual evaluation results.


# 17. Example Analysis

### Example Input

text
The government's disastrous decision has created an
unprecedented crisis and completely failed to protect citizens.


### NLP observations

Potential indicators may include:

text
"disastrous"      → emotionally loaded
"unprecedented"  → strong/exaggerated framing
"completely"     → absolute language
"failed"         → negative evaluative language


### Sentiment

The article is expected to contain strongly negative sentiment.

### Bias Analysis

The system may identify multiple subjective and emotionally loaded expressions.

### LLM Interpretation

The LLM can explain how these expressions may influence reader perception while avoiding the unsupported conclusion that the article is definitively politically biased.


# 18. Limitations

NewsLens has several limitations:

1. Linguistic indicators do not prove that an article is biased.
2. Rule-based detection may produce false positives.
3. Sarcasm and irony can be difficult to interpret.
4. Sentiment models may struggle with mixed or context-dependent sentiment.
5. LLM-generated explanations can contain errors or hallucinations.
6. Text-only analysis cannot evaluate image selection or video framing.
7. The system does not independently verify every factual claim.
8. API availability and latency can affect LLM analysis.
9. Results can depend on the quality and domain of the input article.


# 19. Future Scope

Potential improvements include:

- Training a supervised bias-classification model.
- Building a larger human-annotated news bias dataset.
- Supporting multiple languages.
- Adding source credibility analysis.
- Adding factual claim verification.
- Comparing multiple articles covering the same event.
- Detecting misinformation-related linguistic patterns.
- Adding article URL extraction.
- Adding explainable visualizations.
- Evaluating LLM outputs using a larger human-annotated benchmark.
- Deploying the application as a cloud service.

# 20. Testing

The repository contains automated tests for core components.

Run:

bash
pytest


The tests cover components such as:

- preprocessing;
- sentiment analysis;
- LLM integration.

Additional tests should be added as the system evolves.

# 21. Security

API credentials are stored using environment variables.

The actual `.env` file must not be committed to the repository.

The repository provides:

text
.env.example

as a safe configuration template.


# 22. Conclusion

NewsLens demonstrates how traditional NLP techniques and modern LLM capabilities can be combined into a single practical NLP application.

The system first extracts measurable linguistic information from news articles and then uses an LLM to provide contextual interpretation.

This hybrid architecture combines the consistency and interpretability of traditional NLP components with the contextual language understanding of an LLM.

The system is intended as an analytical and educational tool for identifying potential linguistic patterns and encouraging critical reading, rather than as an absolute authority on whether a news article is biased or unbiased.

#  Author

**Ishansh Agarwal**

GitHub: [@IshanshAg](https://github.com/IshanshAg)

Project: **NewsLens — NLP-Based News Bias & Sentiment Analyzer**
