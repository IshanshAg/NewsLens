from src.sentiment import analyze_vader_sentiment
from src.bias_detector import detect_bias_indicators


def test_vader_sentiment():
    positive_text = "The new policy achieved outstanding success and received widespread praise."
    result = analyze_vader_sentiment(positive_text)
    assert result["label"] == "POSITIVE"
    assert result["compound"] > 0


def test_bias_detection():
    biased_text = "The disastrous policy enacted by the corrupt official will never succeed."
    bias_res = detect_bias_indicators(biased_text)
    
    assert "disastrous" in bias_res["loaded_words_found"]
    assert "never" in bias_res["absolute_terms_found"]
    assert bias_res["bias_score"] > 0