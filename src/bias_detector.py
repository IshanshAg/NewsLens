import logging
from typing import Dict, Any, List, Set
import spacy

logger = logging.getLogger("NewsLens.BiasDetector")
logger.setLevel(logging.DEBUG)

# Categorized Lexicons
LEXICONS = {
    "sensational_terms": {
        "disastrous", "shocking", "monstrous", "outrageous", "devastating", 
        "catastrophic", "reckless", "absurd", "arrogant", "incompetent", 
        "draconian", "appalling", "horrific", "unbelievable", "scandalous"
    },
    "absolute_terms": {
        "always", "never", "completely", "unquestionably", "undeniably", 
        "certainly", "everyone", "nobody", "proves", "prove", "totally", 
        "absolutely", "undoubtedly", "without a doubt"
    },
    "subjective_terms": {
        "incompetent", "arrogant", "terrible", "excellent", "ridiculous", 
        "irresponsible", "disastrous", "monstrous", "foolish", "blatant",
        "out of touch", "disgraceful"
    },
    "loaded_terms": {
        "destroy", "outrage", "failure", "crisis", "alarming", "shocking", 
        "threat", "scheme", "plot", "draconian", "disaster", "bloodbath"
    }
}

def analyze_bias_and_loaded_language(doc: spacy.tokens.Doc) -> Dict[str, Any]:
    """
    Analyzes a spaCy Doc for linguistic bias indicators using lemmatization,
    case-insensitivity, and rule-based lexicon matching.
    """
    raw_text = doc.text
    tokens_text = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

    detected_sensational: Set[str] = set()
    detected_absolute: Set[str] = set()
    detected_subjective: Set[str] = set()
    detected_loaded: Set[str] = set()

    # 1. Token & Lemma Scanning
    for token in doc:
        if token.is_punct or token.is_space:
            continue
            
        t_lower = token.text.lower()
        l_lower = token.lemma_.lower()

        if t_lower in LEXICONS["sensational_terms"] or l_lower in LEXICONS["sensational_terms"]:
            detected_sensational.add(t_lower)
        if t_lower in LEXICONS["absolute_terms"] or l_lower in LEXICONS["absolute_terms"]:
            detected_absolute.add(t_lower)
        if t_lower in LEXICONS["subjective_terms"] or l_lower in LEXICONS["subjective_terms"]:
            detected_subjective.add(t_lower)
        if t_lower in LEXICONS["loaded_terms"] or l_lower in LEXICONS["loaded_terms"]:
            detected_loaded.add(t_lower)

    # 2. Multi-word Phrase Scanning
    text_lower = raw_text.lower()
    if "out of touch" in text_lower:
        detected_subjective.add("out of touch")

    # 3. Calculate Indicators and Density Score
    all_unique_indicators = (
        detected_sensational | detected_absolute | detected_subjective | detected_loaded
    )
    total_indicators = len(all_unique_indicators)
    word_count = max(len(tokens_text), 1)
    
    bias_score = min(round((total_indicators / word_count) * 10, 4), 1.0)

    results = {
        "sensational_terms": sorted(list(detected_sensational)),
        "absolute_terms": sorted(list(detected_absolute)),
        "subjective_terms": sorted(list(detected_subjective)),
        "loaded_terms": sorted(list(detected_loaded)),
        "total_indicators": total_indicators,
        "bias_score": bias_score
    }

    # Debug Logs
    logger.debug(f"RAW TEXT: {raw_text[:80]}...")
    logger.debug(f"DETECTED SENSATIONAL: {results['sensational_terms']}")
    logger.debug(f"DETECTED ABSOLUTE: {results['absolute_terms']}")
    logger.debug(f"DETECTED SUBJECTIVE: {results['subjective_terms']}")
    logger.debug(f"DETECTED LOADED: {results['loaded_terms']}")
    logger.debug(f"TOTAL INDICATORS: {total_indicators} | SCORE: {bias_score}")

    return results