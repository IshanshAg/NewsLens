import pytest
import spacy
from src.bias_detector import analyze_bias_and_loaded_language
from src.sentiment import analyze_sentiment

nlp = spacy.load("en_core_web_sm")

def test_1_clearly_biased_article():
    text = (
        "The government today announced a disastrous economic reform package that will unquestionably "
        "destroy local businesses across the nation. In a shocking display of arrogance, incompetent "
        "officials claimed the draconian policy would boost long-term growth, despite overwhelming outrage. "
        "This monstrous decision proves that politicians are completely out of touch and will never deliver real progress."
    )
    doc = nlp(text)
    res = analyze_bias_and_loaded_language(doc)
    
    assert res["total_indicators"] >= 8
    assert "disastrous" in res["sensational_terms"]
    assert "unquestionably" in res["absolute_terms"]
    assert "incompetent" in res["subjective_terms"]
    assert "destroy" in res["loaded_terms"]

def test_2_neutral_article():
    text = "The committee met on Tuesday to discuss proposed updates to regional zoning regulations."
    doc = nlp(text)
    res = analyze_bias_and_loaded_language(doc)
    
    assert res["total_indicators"] == 0

def test_3_positive_article():
    text = "The local community garden had a successful harvest this quarter, bringing joy to residents."
    doc = nlp(text)
    res = analyze_bias_and_loaded_language(doc)
    
    assert res["total_indicators"] == 0

def test_4_punctuation_and_capitalization():
    text = "DISASTROUS!!! Never!! Shocking, arrogant..."
    doc = nlp(text)
    res = analyze_bias_and_loaded_language(doc)
    
    assert "disastrous" in res["sensational_terms"]
    assert "never" in res["absolute_terms"]

def test_5_word_variation_and_lemmatization():
    text = "The policy destroys businesses, proves ineffective, and is destroying progress."
    doc = nlp(text)
    res = analyze_bias_and_loaded_language(doc)
    
    assert "destroys" in res["loaded_terms"] or "destroying" in res["loaded_terms"]

def test_6_negative_factual_article():
    text = "The earthquake killed hundreds of people and destroyed severe infrastructure in the valley."
    doc = nlp(text)
    bias_res = analyze_bias_and_loaded_language(doc)
    sent_res = analyze_sentiment(text)
    
    assert sent_res["primary_label"] == "NEGATIVE"
    assert "earthquake" not in bias_res["subjective_terms"]