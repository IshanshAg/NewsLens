from typing import Dict, Any
import streamlit as st
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

from src.utils import load_config

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Load configuration
config = load_config()
SENTIMENT_MODEL = config["nlp"].get(
    "sentiment_model", "distilbert-base-uncased-finetuned-sst-2-english"
)
USE_FAST_VADER_ONLY = config["nlp"].get("use_fast_vader_only", False)


@st.cache_resource
def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """
    Instantiates and caches the VADER sentiment analyzer in memory.
    """
    return SentimentIntensityAnalyzer()


@st.cache_resource
def get_transformer_pipeline():
    """
    Caches the HuggingFace transformer model in memory across user sessions.
    Prevents downloading or re-allocating PyTorch memory on every Streamlit rerun.
    """
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        truncation=True,
        max_length=512,
    )


def analyze_vader_sentiment(text: str) -> Dict[str, Any]:
    """
    Rule-based VADER sentiment analysis. Fast and light on CPU.
    """
    if not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "NEUTRAL"}

    vader = get_vader_analyzer()
    scores = vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
        "label": label,
    }


def analyze_transformer_sentiment(text: str) -> Dict[str, Any]:
    """
    Deep learning sentiment analysis using DistilBERT or specified transformer model.
    """
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    sentiment_pipe = get_transformer_pipeline()
    # Truncate text cleanly before passing to transformer
    result = sentiment_pipe(text[:2000])[0]

    return {
        "label": result["label"].upper(),
        "score": round(float(result["score"]), 4),
    }


@st.cache_data(show_spinner=False)
def analyze_sentiment(text: str) -> Dict[str, Any]:
    vader_res = analyze_vader_sentiment(text)
    
    # Fast path: Skip HuggingFace transformer download & CPU inference
    if config["nlp"].get("use_fast_vader_only", False):
        return {
            "vader": vader_res,
            "transformer": {"label": vader_res["label"], "score": abs(vader_res["compound"])},
            "primary_label": vader_res["label"],
            "confidence": abs(vader_res["compound"])
        }

    # Standard path: Deep Learning Transformer
    transformer_res = analyze_transformer_sentiment(text)
    return {
        "vader": vader_res,
        "transformer": transformer_res,
        "primary_label": transformer_res["label"],
        "confidence": transformer_res["score"]
    }