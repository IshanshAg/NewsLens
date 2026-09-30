import pytest
from src.preprocessing import clean_text, extract_linguistic_features, extract_tfidf_features


def test_clean_text():
    raw_html = "<p>The government's <b>disastrous</b> decision... Check http://example.com</p>"
    cleaned = clean_text(raw_html)
    assert "<p>" not in cleaned
    assert "http://example.com" not in cleaned
    assert "The government's disastrous decision..." in cleaned


def test_extract_linguistic_features():
    sample_text = "The disastrous decision made by the incompetent official severely impacted India."
    features = extract_linguistic_features(sample_text)

    assert features["word_count"] > 0
    assert features["sentence_count"] == 1
    assert features["pos_counts"]["adjectives"] >= 2  # 'disastrous', 'incompetent'
    assert any(ent["text"] == "India" for ent in features["entities"])


def test_extract_tfidf_features():
    sample_text = (
        "Economic reform will transform the nation. "
        "The economic policy aims to boost growth across all sectors. "
        "Reforms are necessary for long-term economic stability."
    )
    tfidf = extract_tfidf_features(sample_text, top_n=3)
    assert len(tfidf) > 0
    assert any(item["term"] == "economic" for item in tfidf)