import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from src.utils import load_config, load_sample_articles
from src.preprocessing import extract_linguistic_features, extract_tfidf_features
from src.sentiment import analyze_sentiment
from src.bias_detector import analyze_bias_and_loaded_language
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
# 2. Sidebar Controls & Styled System Architecture Status
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
    st.subheader("⚡ Pipeline Status")

    is_fast_mode = config['nlp'].get('use_fast_vader_only', False)
    fast_mode_label = "Active (Fast VADER)" if is_fast_mode else "Inactive (Full DistilBERT)"
    fast_mode_color = "#10B981" if is_fast_mode else "#6B7280"

    st.markdown(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.05); padding: 14px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); margin-top: 5px;">
            <div style="margin-bottom: 10px;">
                <span style="font-size: 11px; color: #9CA3AF; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">SpaCy Model</span><br>
                <code style="background: rgba(0,0,0,0.3); color: #60A5FA; padding: 3px 8px; border-radius: 5px; font-size: 13px;">{config['nlp']['spacy_model']}</code>
            </div>
            <div style="margin-bottom: 10px;">
                <span style="font-size: 11px; color: #9CA3AF; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Sentiment Engine</span><br>
                <code style="background: rgba(0,0,0,0.3); color: #A78BFA; padding: 3px 8px; border-radius: 5px; font-size: 12px;">{config['nlp']['sentiment_model']}</code>
            </div>
            <div>
                <span style="font-size: 11px; color: #9CA3AF; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">VADER Fast Mode</span><br>
                <span style="color: {fast_mode_color}; font-weight: bold; font-size: 13px;">● {fast_mode_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
# 4. Pipeline Execution
# -----------------------------------------------------------------------------
if analyze_btn:
    if not article_text.strip():
        st.warning("Please provide valid article text before running the analysis.")
    else:
        with st.spinner("⚡ Running NLP pipelines and LLM inference..."):
            # Step 1: Preprocessing & Fast Deterministic Feature Extraction
            ling_features = extract_linguistic_features(article_text)
            doc = ling_features.get("doc")  # Reuse spaCy Doc object directly
            
            if doc is not None:
                bias_res = analyze_bias_and_loaded_language(doc)
            else:
                import spacy
                nlp = spacy.load(config['nlp']['spacy_model'])
                bias_res = analyze_bias_and_loaded_language(nlp(article_text))

            tfidf_terms = extract_tfidf_features(article_text)

            # Step 2: Sentiment Inference
            sentiment_res = analyze_sentiment(article_text)

            # Step 3: LLM Contextual Synthesis
            llm_explanation = generate_llm_analysis(
                article_text,
                ling_features,
                sentiment_res,
                bias_res,
            )

        st.success("Analysis complete!")

        # -----------------------------------------------------------------------------
        # 5. Output Visualization Dashboard
        # -----------------------------------------------------------------------------
        # Top-level key metrics banner
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Word Count", ling_features.get("word_count", 0))
        mcol2.metric("Sentences", ling_features.get("sentence_count", 0))
        
        confidence_val = sentiment_res.get("confidence")
        conf_display = f"{confidence_val:.2f} score" if confidence_val is not None else "N/A"
        
        mcol3.metric(
            "Primary Sentiment",
            sentiment_res.get("primary_label", "N/A"),
            delta=conf_display,
        )
        
        mcol4.metric(
            "Linguistic Indicators",
            bias_res.get("total_indicators", 0),
            delta=f"Density: {bias_res.get('bias_score', 0.0)}",
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
            st.subheader("Identified Subjective & Loaded Language Categories")
            bcol1, bcol2 = st.columns(2)
            
            with bcol1:
                st.markdown("##### 🚨 Sensational / Hyperbolic")
                sensational = bias_res.get("sensational_terms", [])
                if sensational:
                    for w in sensational:
                        st.markdown(f"- `{w}`")
                else:
                    st.caption("No explicit sensational terminology flagged.")

                st.markdown("##### ⚖️ Subjective / Evaluative")
                subjective = bias_res.get("subjective_terms", [])
                if subjective:
                    for w in subjective:
                        st.markdown(f"- `{w}`")
                else:
                    st.caption("No subjective bias indicators flagged.")

            with bcol2:
                st.markdown("##### 📌 Absolute / Certainty Language")
                absolute = bias_res.get("absolute_terms", [])
                if absolute:
                    for w in absolute:
                        st.markdown(f"- `{w}`")
                else:
                    st.caption("No absolute certainty terms flagged.")

                st.markdown("##### 💥 Loaded / Emotional Terms")
                loaded = bias_res.get("loaded_terms", [])
                if loaded:
                    for w in loaded:
                        st.markdown(f"- `{w}`")
                else:
                    st.caption("No strong emotional loaded terms flagged.")

            st.markdown("---")
            with st.expander("🔍 Developer Pipeline Debug Log"):
                st.json(bias_res)

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