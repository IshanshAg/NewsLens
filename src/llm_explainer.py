import os
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def get_gemini_client():
    """
    Safely retrieves the Gemini API key from environment variables or Streamlit secrets
    and initializes the Google GenAI client.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key or not str(api_key).strip():
        return None

    try:
        return genai.Client(api_key=str(api_key).strip())
    except Exception:
        return None


def generate_llm_analysis(
    text: str,
    ling_features: Dict[str, Any],
    sentiment_res: Dict[str, Any],
    bias_res: Dict[str, Any],
) -> str:
    """
    Generates structured framing and bias analysis using Google's Gemini API.
    """
    client = get_gemini_client()

    if client is None:
        return (
            "⚠️ **Gemini API Key Missing**\n\n"
            "To enable automated LLM summaries, please add your key to your `.env` file:\n"
            "```env\n"
            'GEMINI_API_KEY="AIzaSy_your_key_here"\n'
            "```\n"
            "Get a free key from [Google AI Studio](https://aistudio.google.com/)."
        )

    # Extract NLP & Bias context from pipeline dictionary output
    word_count = ling_features.get("word_count", len(text.split()))
    sentiment_label = sentiment_res.get("primary_label", "NEUTRAL")
    vader_compound = sentiment_res.get("vader_compound", 0.0)

    sensational = bias_res.get("sensational_terms", [])
    absolute = bias_res.get("absolute_terms", [])
    subjective = bias_res.get("subjective_terms", [])
    loaded = bias_res.get("loaded_terms", [])

    prompt = f"""
You are an expert media literacy analyst evaluating a news article snippet for framing, stance, and linguistic bias indicators.

Article Snippet:
"{text[:2000]}"

Rule-Based NLP Metrics Context:
- Word Count: {word_count}
- Primary Sentiment: {sentiment_label} (VADER Compound: {vader_compound})
- Rule-Based Sensational Terms: {', '.join(sensational) if sensational else 'None'}
- Rule-Based Absolute Terms: {', '.join(absolute) if absolute else 'None'}
- Rule-Based Subjective Terms: {', '.join(subjective) if subjective else 'None'}
- Rule-Based Loaded Terms: {', '.join(loaded) if loaded else 'None'}

ANALYSIS GUIDELINES:
1. Do NOT state that the article is politically or objectively "biased". Use neutral diagnostic terminology such as "potential linguistic bias indicators", "loaded language", "subjective framing", or "absolute/certainty language".
2. Treat rule-based indicators as baseline evidence. If you notice additional charged or manipulative phrasing in the article snippet that was missed by the rule-based detector, explicitly list them in a dedicated subsection titled "**LLM-Identified Additional Terms**".
3. Clearly distinguish between rule-based detected terms and LLM-identified additional terms.

Provide your response strictly following these three section headers (do not output any introductory fluff or chat responses before the first header):

### 1. Executive Framing & Stance
Analyze the main perspective or narrative angle prioritized by the article.

### 2. Linguistic Indicators Breakdown
Analyze how loaded words, emotional tone, and absolute phrasing shape reader perception. Include the comparison between rule-based detected terms and any additional terms identified by the LLM.

### 3. Critical Media Literacy Recommendation
Provide actionable guidance on how a critical reader should evaluate this coverage.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a precise, neutral media literacy diagnostic tool.",
                temperature=0.3,
                max_output_tokens=1500,
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ **LLM Generation Failed**: {str(e)}"