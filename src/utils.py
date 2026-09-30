import os
import yaml
from typing import Dict, Any

# Resolve absolute base path to locate config and data directories reliably
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "sample_articles")


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads YAML configuration parameters.
    Returns default settings if the configuration file is missing.
    """
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # Default fallback parameters
    return {
        "nlp": {
            "spacy_model": "en_core_web_sm",
            "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
            "use_fast_vader_only": True,
        },
        "analysis": {"top_tf_idf_terms": 10},
    }


def load_sample_articles() -> Dict[str, str]:
    """
    Loads sample txt files from data/sample_articles directory,
    or provides built-in fallback articles if the directory doesn't exist yet.
    """
    samples = {}

    # Option 1: Load from directory if files exist
    if os.path.exists(SAMPLES_DIR):
        for filename in os.listdir(SAMPLES_DIR):
            if filename.endswith(".txt"):
                article_title = filename.replace(".txt", "").replace("_", " ").title()
                filepath = os.path.join(SAMPLES_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        samples[article_title] = f.read().strip()
                except Exception:
                    continue

    # Option 2: Fallback sample articles if folder is empty or missing
    if not samples:
        samples = {
            "Tech Regulation Debate": (
                "Lawmakers announced unprecedented regulatory proposals targeting major technology firms today. "
                "Critics argue the massive restrictions will paralyze innovation and devastate emerging startups, "
                "while proponents claim radical oversight is urgently needed to curb market dominance."
            ),
            "Economic Growth Report": (
                "The quarterly financial report highlighted a steady rise in employment rates and consumer confidence. "
                "Economists praised the strategic monetary policies, pointing toward long-term stabilization, "
                "though persistent inflation figures continue to present minor headwinds."
            ),
        }

    return samples