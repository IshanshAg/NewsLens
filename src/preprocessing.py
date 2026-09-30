import re
from typing import Dict, List, Any
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import load_config

# Load system configuration
config = load_config()
SPACY_MODEL = config["nlp"]["spacy_model"]
TOP_TFIDF_TERMS = config["analysis"].get("top_tf_idf_terms", 10)


@st.cache_resource
def load_spacy_model():
    """
    Loads and caches the spaCy language model pipeline in memory.
    This prevents re-loading the model binaries on every Streamlit rerun.
    """
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        raise ImportError(
            f"spaCy model '{SPACY_MODEL}' not found. "
            f"Run 'python -m spacy download {SPACY_MODEL}' before running the application."
        )


# Global cached instance
nlp = load_spacy_model()


def clean_text(text: str) -> str:
    """
    Cleans raw input text by removing HTML tags, URLs, extra whitespace, 
    and non-standard Unicode characters.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http[s]?://\S+", " ", text)
    # Normalize multiple whitespace, tabs, and newlines
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


@st.cache_data(show_spinner=False)
def extract_linguistic_features(text: str) -> Dict[str, Any]:
    """
    Processes cleaned text through spaCy to extract lemmatized tokens,
    POS frequencies, and Named Entities (NER). Cached by Streamlit for identical inputs.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {
            "cleaned_text": "",
            "tokens": [],
            "lemmas": [],
            "pos_counts": {},
            "entities": [],
            "sentence_count": 0,
            "word_count": 0,
            "avg_sentence_length": 0.0,
        }

    doc = nlp(cleaned_text)

    # Filter out punctuation, whitespace, and stop words for lemmatization
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

    # Calculate POS Counts (Adjectives, Adverbs, Verbs, Nouns)
    pos_counts = {
        "adjectives": sum(1 for token in doc if token.pos_ == "ADJ"),
        "adverbs": sum(1 for token in doc if token.pos_ == "ADV"),
        "verbs": sum(1 for token in doc if token.pos_ == "VERB"),
        "nouns": sum(1 for token in doc if token.pos_ == "NOUN"),
        "proper_nouns": sum(1 for token in doc if token.pos_ == "PROPN"),
    }

    # Named Entity Recognition (NER)
    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "EVENT", "LAW"]
    ]

    # Structural metrics
    sentences = list(doc.sents)
    sentence_count = len(sentences)
    word_count = len(tokens)
    avg_sentence_length = round(word_count / sentence_count, 2) if sentence_count > 0 else 0.0

    return {
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_counts": pos_counts,
        "entities": entities,
        "sentence_count": sentence_count,
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
    }


@st.cache_data(show_spinner=False)
def extract_tfidf_features(text: str, top_n: int = TOP_TFIDF_TERMS) -> List[Dict[str, Any]]:
    """
    Extracts top TF-IDF keywords and their weights from the text.
    Cached by Streamlit for identical inputs.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []

    # Use spaCy lemmas as sentence corpus to prevent root word duplication
    doc = nlp(cleaned_text)
    sentence_corpus = [
        " ".join([token.lemma_.lower() for token in sent if not token.is_stop and token.is_alpha])
        for sent in doc.sents
    ]

    # Filter out empty sentences
    sentence_corpus = [sent for sent in sentence_corpus if len(sent.split()) > 1]

    if not sentence_corpus:
        return []

    try:
        vectorizer = TfidfVectorizer(max_features=top_n, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentence_corpus)
        
        # Aggregate mean TF-IDF score across all sentences
        mean_scores = tfidf_matrix.mean(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()

        # Sort features by weight
        top_terms = sorted(
            [{"term": feature_names[i], "score": round(float(mean_scores[i]), 4)} for i in range(len(feature_names))],
            key=lambda x: x["score"],
            reverse=True
        )
        return top_terms[:top_n]
    except ValueError:
        # Occurs if all words are stop words or vocabulary is empty
        return []