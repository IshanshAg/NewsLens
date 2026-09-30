import re
from typing import Dict, List, Any
import spacy
from src.preprocessing import extract_linguistic_features, clean_text

nlp = spacy.load("en_core_web_sm")

# Rule-based linguistic lexicons for bias detection
LOADED_WORDS = {
    "disastrous", "incompetent", "outrageous", "corrupt", "unprecedented",
    "scandalous", "catastrophic", "shocking", "draconian", "miraculous",
    "tyrannical", "heroic", "devastating", "monstrous", "blatant"
}

ABSOLUTE_TERMS = {
    "always", "never", "everyone", "nobody", "completely", "undeniably",
    "unquestionably", "totally", "absolutely", "certainly", "impossible"
}


def detect_bias_indicators(text: str) -> Dict[str, Any]:
    """
    Scans the text for loaded terminology, absolute statements, subjective qualifiers,
    and calculates an overall linguistic bias score index (0 to 100).
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "loaded_words_found": [],
            "absolute_terms_found": [],
            "subjective_adjectives": [],
            "bias_score": 0.0,
            "counts": {"loaded": 0, "absolutes": 0, "subjective": 0}
        }

    doc = nlp(cleaned)
    tokens_lower = [token.text.lower() for token in doc if not token.is_punct]
    word_count = max(len(tokens_lower), 1)

    # 1. Detect Loaded Words
    loaded_found = list(set([word for word in tokens_lower if word in LOADED_WORDS]))

    # 2. Detect Absolute Statements
    absolutes_found = list(set([word for word in tokens_lower if word in ABSOLUTE_TERMS]))

    # 3. Detect Subjective Adjectives (Adjectives not used purely as classifiers)
    subjective_adj = list(set([
        token.text.lower() for token in doc 
        if token.pos_ == "ADJ" and token.text.lower() not in loaded_found
        and len(token.text) > 3
    ]))[:10]  # Cap top 10

    # 4. Calculate Quantified Bias Score Index
    # Density metric normalized per 100 words
    loaded_density = (len(loaded_found) / word_count) * 100
    absolute_density = (len(absolutes_found) / word_count) * 100
    subjective_density = (len(subjective_adj) / word_count) * 100

    raw_score = (loaded_density * 35.0) + (absolute_density * 25.0) + (subjective_density * 15.0)
    bias_score = min(round(raw_score, 2), 100.0)

    return {
        "loaded_words_found": loaded_found,
        "absolute_terms_found": absolutes_found,
        "subjective_adjectives": subjective_adj,
        "bias_score": bias_score,
        "counts": {
            "loaded": len(loaded_found),
            "absolutes": len(absolutes_found),
            "subjective": len(subjective_adj)
        }
    }