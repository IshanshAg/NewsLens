from src.llm_analyzer import build_pipeline_payload


def test_build_pipeline_payload():
    sample_text = "The government announced a new economic reform."
    ling_feats = {"word_count": 7, "sentence_count": 1, "avg_sentence_length": 7.0, "pos_counts": {}, "entities": []}
    sent_res = {"primary_label": "POSITIVE", "confidence": 0.85, "vader": {}, "transformer": {}}
    bias_res = {"bias_score": 12.5, "loaded_words_found": ["reform"], "absolute_terms_found": [], "subjective_adjectives": []}

    payload = build_pipeline_payload(sample_text, ling_feats, sent_res, bias_res)
    
    assert "article_text" in payload
    assert "bias_score_index" in payload
    assert "POSITIVE" in payload