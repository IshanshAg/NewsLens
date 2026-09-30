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

    # Extract features for prompt context
    word_count = ling_features.get("word_count", 0)
    sentiment_label = sentiment_res.get("primary_label", "NEUTRAL")
    sensational_words = bias_res.get("sensational_words", [])
    bias_words = bias_res.get("bias_words", [])

    prompt = f"""
Analyze the following news article snippet for framing and bias using the provided NLP context.

Article Snippet:
"{text[:2000]}"

NLP Metrics Context:
- Word Count: {word_count}
- Primary Sentiment: {sentiment_label}
- Sensational Terms: {', '.join(sensational_words) if sensational_words else 'None'}
- Loaded/Bias Terms: {', '.join(bias_words) if bias_words else 'None'}

Provide a thorough analysis strictly following these three section headers (do not include introductory or conversational fluff before the first header):

### 1. Executive Framing Summary
[Explain the primary narrative angle or perspective prioritized]

### 2. Linguistic Stance & Tone
[Analyze how loaded words, emotional tone, and sentence structure shape reader perception]

### 3. Media Literacy Recommendation
[Give actionable guidance on how a critical reader should evaluate this coverage]
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