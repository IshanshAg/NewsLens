import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from src.utils import load_config, load_sample_articles
from src.preprocessing import extract_linguistic_features, extract_tfidf_features
from src.sentiment import analyze_sentiment
from src.bias_detector import detect_bias_indicators
from src.llm_explainer import generate_llm_analysis

# -----------------------------------------------------------------------------
# 1. Page Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NewsLens - Media Bias & Sentiment Analysis",
    page_icon="📰",
    layout="wide",
)

config = load_config()
sample_articles = load_sample_articles()

st.title("📰 NewsLens: Media Bias & Sentiment Analyzer")
st.caption(
    "Automated NLP diagnostic tool for linguistic framing, sentiment polarity, and cognitive bias detection."
)

# -----------------------------------------------------------------------------
# 2. Sidebar Controls & Sample Article Loader
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration & Inputs")
    
    sample_choice = st.selectbox(
        "Load Sample Article",
        options=["Custom Text"] + list(sample_articles.keys()),
        index=0,
    )
    
    selected_text = ""
    if sample_choice != "Custom Text":
        selected_text = sample_articles[sample_choice]

    st.markdown("---")
    st.caption("⚙️ **Pipeline Architecture**")

    st.markdown(f"**SpaCy Model:** `:gray-badge[{config['nlp']['spacy_model']}]`")
    st.markdown(f"**Sentiment Engine:** `:blue-badge[{config['nlp']['sentiment_model']}]`")

    fast_mode_status = "Enabled ⚡" if config['nlp'].get('use_fast_vader_only') else "Disabled 🐢"
    badge_color = "green" if config['nlp'].get('use_fast_vader_only') else "gray"
    st.markdown(f"**Fast Mode:** `:{badge_color}-badge[{fast_mode_status}]`")

# -----------------------------------------------------------------------------
# 3. Main Text Input Form
# -----------------------------------------------------------------------------
initial_text = selected_text if selected_text else ""
article_text = st.text_area(
    "Paste News Article Text Below:",
    value=initial_text,
    height=250,
    placeholder="Enter the full article content here to analyze linguistic framing and sentiment...",
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze_btn = st.button("🚀 Analyze Article", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Concurrent Analysis Pipeline Execution
# -----------------------------------------------------------------------------
if analyze_btn:
    if not article_text.strip():
        st.warning("Please provide valid article text before running the analysis.")
    else:
        with st.spinner("⚡ Running concurrent NLP pipelines and LLM inference..."):
            # Step 1: Fast local deterministic features (CPU fast-path)
            ling_features = extract_linguistic_features(article_text)
            bias_res = detect_bias_indicators(article_text)
            tfidf_terms = extract_tfidf_features(article_text)

            # Step 2: Parallelize Heavy Model Inference (DistilBERT Local + OpenAI Network Call)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_sentiment = executor.submit(analyze_sentiment, article_text)
                
                # Placeholder sentiment for prompt context generation
                sent_placeholder = {"primary_label": "PROCESSING", "confidence": 0.0}
                future_llm = executor.submit(
                    generate_llm_analysis,
                    article_text,
                    ling_features,
                    sent_placeholder,
                    bias_res,
                )

                # Collect concurrent execution outputs
                sentiment_res = future_sentiment.result()
                llm_explanation = future_llm.result()

        st.success("Analysis complete!")

        # -----------------------------------------------------------------------------
        # 5. Output Visualization Dashboard
        # -----------------------------------------------------------------------------
        # Top-level key metrics banner
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Word Count", ling_features.get("word_count", 0))
        mcol2.metric("Sentences", ling_features.get("sentence_count", 0))
        mcol3.metric(
            "Primary Sentiment",
            sentiment_res.get("primary_label", "N/A"),
            delta=f"{sentiment_res.get('confidence', 0.0):.2f} score",
        )
        mcol4.metric(
            "Loaded Words Detected",
            len(bias_res.get("sensational_words", [])) + len(bias_res.get("bias_words", [])),
        )

        st.markdown("---")

        # Tabbed details layout
        tab_llm, tab_sentiment, tab_bias, tab_linguistics = st.tabs(
            [
                "🤖 LLM Analysis & Synthesis",
                "🎭 Sentiment Breakdown",
                "⚠️ Bias & Loaded Language",
                "📊 Linguistic Metrics",
            ]
        )

        # Tab 1: LLM Explainer
        with tab_llm:
            st.subheader("Executive Framing & Bias Summary")
            st.markdown(llm_explanation)

        # Tab 2: Sentiment Comparison
        with tab_sentiment:
            st.subheader("Dual Model Sentiment Diagnostics")
            scol1, scol2 = st.columns(2)
            
            with scol1:
                st.markdown("#### Rule-Based VADER Polarity")
                vader_data = sentiment_res.get("vader", {})
                st.json(vader_data)

            with scol2:
                st.markdown("#### Transformer (DistilBERT) Classification")
                trans_data = sentiment_res.get("transformer", {})
                st.json(trans_data)

        # Tab 3: Bias Indicators
        with tab_bias:
            st.subheader("Identified Subjective & Sensational Language")
            bcol1, bcol2 = st.columns(2)
            
            with bcol1:
                st.markdown("##### Sensational/Hyperbolic Words")
                sensational = bias_res.get("sensational_words", [])
                if sensational:
                    st.write(", ".join([f"`{w}`" for w in sensational]))
                else:
                    st.info("No explicit sensational terminology detected.")

            with bcol2:
                st.markdown("##### Bias / Subjective Terminology")
                bias_words = bias_res.get("bias_words", [])
                if bias_words:
                    st.write(", ".join([f"`{w}`" for w in bias_words]))
                else:
                    st.info("No subjective bias indicators flagged.")

        # Tab 4: Linguistic Features & TF-IDF
        with tab_linguistics:
            st.subheader("Structural & Keyword Extraction")
            lcol1, lcol2 = st.columns(2)

            with lcol1:
                st.markdown("##### Part-of-Speech Counts")
                st.json(ling_features.get("pos_counts", {}))

            with lcol2:
                st.markdown("##### Key TF-IDF Terms")
                if tfidf_terms:
                    for item in tfidf_terms:
                        st.progress(
                            min(float(item["score"]), 1.0),
                            text=f"**{item['term']}** (score: {item['score']})",
                        )
                else:
                    st.info("No significant TF-IDF keywords extracted.")