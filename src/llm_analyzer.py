import json
from typing import Dict, Any
from openai import OpenAI
from src.utils import load_config, load_prompt_template, get_env_variable

# Load system configuration
config = load_config()
MODEL_NAME = config["llm"]["model"]
TEMPERATURE = config["llm"]["temperature"]
MAX_TOKENS = config["llm"]["max_tokens"]


def build_pipeline_payload(
    article_text: str,
    linguistic_features: Dict[str, Any],
    sentiment_results: Dict[str, Any],
    bias_results: Dict[str, Any]
) -> str:
    """
    Formats the raw outputs from upstream NLP modules into a clean, 
    structured JSON string for the LLM prompt.
    """
    payload = {
        "article_text": article_text,
        "metrics": {
            "word_count": linguistic_features.get("word_count", 0),
            "sentence_count": linguistic_features.get("sentence_count", 0),
            "avg_sentence_length": linguistic_features.get("avg_sentence_length", 0.0),
            "pos_counts": linguistic_features.get("pos_counts", {})
        },
        "sentiment_analysis": {
            "vader_scores": sentiment_results.get("vader", {}),
            "transformer": sentiment_results.get("transformer", {}),
            "primary_label": sentiment_results.get("primary_label", "NEUTRAL"),
            "confidence": sentiment_results.get("confidence", 0.0)
        },
        "bias_indicators": {
            "bias_score_index": bias_results.get("bias_score", 0.0),
            "loaded_words": bias_results.get("loaded_words_found", []),
            "absolute_terms": bias_results.get("absolute_terms_found", []),
            "subjective_adjectives": bias_results.get("subjective_adjectives", [])
        },
        "named_entities": linguistic_features.get("entities", [])
    }
    
    return json.dumps(payload, indent=2)


def generate_llm_analysis(
    article_text: str,
    linguistic_features: Dict[str, Any],
    sentiment_results: Dict[str, Any],
    bias_results: Dict[str, Any]
) -> str:
    """
    Sends structured NLP results to OpenAI API and retrieves contextual 
    explanation and neutral summary.
    """
    api_key = get_env_variable("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # 1. Prepare structured payload
    payload_json = build_pipeline_payload(
        article_text, linguistic_features, sentiment_results, bias_results
    )

    # 2. Populate prompt template
    prompt_template = load_prompt_template("prompts/news_analysis.txt")
    full_prompt = prompt_template.replace("{pipeline_payload}", payload_json)

    # 3. Call OpenAI Chat Completions API
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert NLP news analysis assistant. "
                        "Evaluate the text objectively using the provided pipeline features."
                    )
                },
                {"role": "user", "content": full_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error executing LLM analysis: {str(e)}"