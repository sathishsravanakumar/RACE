"""
RACE: RAG-Optimized Clinical Reasoning Engine
Streamlit Web Interface - Professional Medical Design
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import ClinicalReasoningEngine

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="RACE - Clinical Reasoning Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CACHED MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_reasoning_engine():
    """
    Load the Clinical Reasoning Engine with caching.
    This ensures the model is only loaded once and reused across sessions.
    """
    try:
        with st.spinner("Loading Clinical Reasoning Engine..."):
            engine = ClinicalReasoningEngine()
        return engine, None
    except Exception as e:
        return None, str(e)


# =============================================================================
# CUSTOM CSS - PROFESSIONAL MEDICAL DESIGN
# =============================================================================

def load_custom_css():
    """Apply professional medical website styling."""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global Styles - Dark Theme with White Text */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #ffffff !important;
        }

        .main {
            background: #1a1a1a;
            font-family: 'Inter', sans-serif;
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Hero Section - Clean Gradient */
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.25);
            color: #ffffff;
            text-align: center;
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            color: #ffffff;
            letter-spacing: -0.5px;
        }

        .hero-subtitle {
            font-size: 1.2rem;
            font-weight: 400;
            max-width: 800px;
            margin: 0 auto;
            color: rgba(255, 255, 255, 0.95);
            line-height: 1.6;
        }

        /* Feature Cards - Dark with White Text */
        .feature-card {
            background: #2d2d2d;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            margin: 0.5rem;
            transition: all 0.3s ease;
            border: 1px solid #404040;
            height: 100%;
            min-height: 200px;
            display: flex;
            flex-direction: column;
        }

        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            border-color: #667eea;
        }

        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        .feature-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff !important;
            margin-bottom: 0.75rem;
            letter-spacing: -0.3px;
        }

        .feature-text {
            color: #e0e0e0 !important;
            line-height: 1.6;
            font-weight: 400;
            font-size: 0.95rem;
        }

        /* Input Section - Dark */
        .input-section {
            background: #2d2d2d;
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            margin: 2rem 0;
            border: 1px solid #404040;
        }

        .section-header {
            font-size: 1.75rem;
            font-weight: 700;
            color: #ffffff !important;
            margin-bottom: 1.5rem;
            text-align: center;
            letter-spacing: -0.5px;
        }

        /* Text Area Custom Styling - White Text */
        .stTextArea textarea {
            border-radius: 8px;
            border: 2px solid #404040;
            font-size: 1rem;
            padding: 1rem;
            transition: all 0.3s ease;
            background: #1a1a1a;
            color: #ffffff !important;
            font-weight: 400;
            line-height: 1.6;
        }

        .stTextArea textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
            background: #1a1a1a;
            outline: none;
        }

        .stTextArea label {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        /* Button Styling - Modern Gradient */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff !important;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.875rem 2.5rem;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            letter-spacing: 0.3px;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }

        .stButton > button p {
            color: #ffffff !important;
            font-weight: 600;
        }

        /* Results Section - Dark */
        .results-container {
            background: #2d2d2d;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            margin-top: 2rem;
            border: 1px solid #404040;
        }

        .result-column-header {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid #667eea;
            color: #ffffff !important;
            letter-spacing: -0.3px;
        }

        /* Evidence Boxes - Blue Dark */
        .evidence-box {
            background: #1e3a5f;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
        }

        .evidence-title {
            font-weight: 700;
            color: #60a5fa !important;
            margin-bottom: 0.75rem;
            font-size: 1.05rem;
        }

        .evidence-content {
            color: #ffffff !important;
            line-height: 1.7;
            font-size: 0.95rem;
            font-weight: 400;
        }

        /* Reasoning Box - Green Dark */
        .reasoning-box {
            background: #1a3d2e;
            padding: 2rem;
            border-radius: 10px;
            border-left: 4px solid #10b981;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);
        }

        .reasoning-content {
            color: #ffffff !important;
            line-height: 1.8;
            font-size: 1rem;
            font-weight: 400;
        }

        .reasoning-content ul, .reasoning-content ol {
            margin: 0.75rem 0;
            padding-left: 1.5rem;
        }

        .reasoning-content li {
            margin: 0.5rem 0;
            color: #ffffff !important;
        }

        .reasoning-content strong {
            color: #10b981 !important;
            font-weight: 700;
        }

        /* Download Button - Green */
        .stDownloadButton > button {
            background: #10b981;
            color: #ffffff;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);
            transition: all 0.3s ease;
        }

        .stDownloadButton > button:hover {
            background: #059669;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        }

        /* Settings Panel - Dark */
        .settings-panel {
            background: #2d2d2d;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            margin-bottom: 1.5rem;
            border: 1px solid #404040;
        }

        .settings-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #ffffff !important;
            margin-bottom: 1rem;
            letter-spacing: -0.3px;
        }

        /* Status Badge - Green */
        .status-badge {
            display: inline-block;
            background: #10b981;
            color: #ffffff;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.875rem;
            margin: 0.5rem 0;
        }

        /* Example Questions - Dark */
        .example-box {
            background: #2d2d2d;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #404040;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
            color: #ffffff !important;
            font-weight: 400;
        }

        .example-box:hover {
            background: #3d3d3d;
            border-color: #667eea;
            transform: translateX(3px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        }

        /* Expander Styling - Dark */
        .streamlit-expanderHeader {
            background: #2d2d2d;
            border-radius: 8px;
            font-weight: 600;
            color: #ffffff !important;
            border: 1px solid #404040;
        }

        /* Slider Styling */
        .stSlider > div > div > div {
            color: #ffffff;
        }

        /* Sidebar Styling - Dark */
        section[data-testid="stSidebar"] {
            background: #2d2d2d;
            border-right: 1px solid #404040;
        }

        /* Fix all text visibility - WHITE TEXT */
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color: #ffffff !important;
        }

        /* Expander content */
        .streamlit-expanderContent {
            background: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 1rem;
            color: #ffffff !important;
        }

        /* Info messages */
        .stInfo {
            background: #1e3a5f;
            color: #ffffff !important;
            border-left: 4px solid #3b82f6;
        }

        /* Warning messages */
        .stWarning {
            background: #4a3800;
            color: #ffffff !important;
            border-left: 4px solid #f59e0b;
        }

        /* Error messages */
        .stError {
            background: #4a1414;
            color: #ffffff !important;
            border-left: 4px solid #ef4444;
        }

        /* Success messages */
        .stSuccess {
            background: #1a3d2e;
            color: #ffffff !important;
            border-left: 4px solid #10b981;
        }

        /* Spinner text */
        .stSpinner > div {
            color: #ffffff !important;
        }

        /* All paragraph text - WHITE */
        p {
            color: #ffffff !important;
        }

        /* All div text - WHITE */
        div {
            color: #ffffff !important;
        }

        /* Label text - WHITE */
        label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Ensure code blocks are visible */
        code {
            background: #2d2d2d;
            color: #ffffff !important;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }

        /* Column alignment */
        [data-testid="column"] {
            display: flex;
            flex-direction: column;
        }

        /* Streamlit containers */
        .element-container {
            width: 100%;
        }

        /* Fix slider styling */
        .stSlider {
            padding: 0.5rem 0;
        }

        /* Row alignment */
        .row-widget {
            display: flex;
            align-items: stretch;
        }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application logic with professional medical design."""

    # Apply custom CSS
    load_custom_css()

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">🏥 RACE</div>
            <div class="hero-subtitle">
                RAG-Optimized Clinical Reasoning Engine<br>
                Making Health Care Better Together with AI-Powered Evidence-Based Medicine
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Feature Cards Row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📚</div>
                <div class="feature-title">Evidence Retrieval</div>
                <div class="feature-text">
                    Advanced RAG system retrieves relevant medical guidelines from comprehensive knowledge base
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">AI Reasoning</div>
                <div class="feature-text">
                    Llama-3-8B powered clinical reasoning with chain-of-thought analysis. Provides step-by-step medical reasoning grounded in retrieved evidence for transparent, explainable clinical decision support.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Fast & Accurate</div>
                <div class="feature-text">
                    4-bit quantized model delivers rapid responses with medical-grade accuracy
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Load model
    engine, error = load_reasoning_engine()

    # Sidebar Info
    with st.sidebar:
        # Default settings values
        temperature = 0.7
        max_tokens = 128
        top_k_retrieval = 2
        st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
        st.markdown('<div class="settings-title">⚙️ Model Settings</div>', unsafe_allow_html=True)

        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )

        max_tokens = st.slider(
            "📏 Max Response Length",
            min_value=64,
            max_value=256,
            value=128,
            step=32,
            help="Maximum length of AI response (shorter = more concise)"
        )

        top_k_retrieval = st.slider(
            "📊 Evidence Chunks",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Number of evidence sources to retrieve"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        # System Status
        st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
        st.markdown('<div class="settings-title">📊 System Status</div>', unsafe_allow_html=True)
        if engine:
            st.markdown('<span class="status-badge">✓ Model Ready</span>', unsafe_allow_html=True)
            st.info(f"📍 {engine.vectorstore._collection.count()} medical documents loaded")
        else:
            st.error("❌ Model initialization failed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # About
        with st.expander("ℹ️ About RACE"):
            st.markdown("""
            **RACE** combines cutting-edge AI technologies:

            - 📚 **RAG**: Retrieval-Augmented Generation
            - 🧠 **LLM**: Llama-3-8B with QLoRA fine-tuning
            - ⚡ **Optimization**: 4-bit quantization
            - 🎯 **Purpose**: Evidence-based clinical reasoning

            *For educational and research purposes.*
            """)

    # Main Content
    if not engine:
        st.error(f"⚠️ Failed to initialize Clinical Reasoning Engine: {error}")
        st.info("💡 Please run `python src/ingest.py` to set up the knowledge base")
        return

    # Input Section
    st.markdown('<div class="section-header" style="margin-top: 2rem;">🩺 Enter Clinical Scenario</div>', unsafe_allow_html=True)

    # Example questions
    with st.expander("💡 Example Clinical Questions"):
        examples = [
            "What is the recommended starting dose of metformin?",
            "What are the contraindications for metformin therapy?",
            "When should insulin therapy be initiated in type 2 diabetes?",
            "What is the role of GLP-1 receptor agonists in diabetes management?",
            "What are first-line antihypertensive agents for Black patients?",
            "When is anticoagulation indicated in atrial fibrillation?",
            "What medications reduce mortality in HFrEF patients?",
            "How should statin therapy be monitored?"
        ]
        for ex in examples:
            st.markdown(f'<div class="example-box">• {ex}</div>', unsafe_allow_html=True)

    # Query input
    query = st.text_area(
        "Enter your clinical question or scenario:",
        height=120,
        placeholder="Example: A 55-year-old patient with newly diagnosed type 2 diabetes (HbA1c 8.5%) and normal renal function. What is the recommended first-line treatment?",
        label_visibility="collapsed"
    )

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Clinical Scenario", use_container_width=True, type="primary")

    # Process query
    if analyze_button:
        if not query.strip():
            st.warning("⚠️ Please enter a clinical scenario or question")
            return

        # Processing
        with st.spinner("🔄 Analyzing clinical scenario with AI..."):
            try:
                result = engine.generate_response(
                    query=query,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k_retrieval
                )

                # Results Section
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📋 Analysis Results</div>', unsafe_allow_html=True)

                # Two-column layout
                col_evidence, col_reasoning = st.columns([1, 1])

                # Left Column - Retrieved Evidence
                with col_evidence:
                    st.markdown('<div class="result-column-header">📚 Retrieved Medical Evidence</div>', unsafe_allow_html=True)

                    retrieved_docs = result["retrieved_context"]

                    if retrieved_docs:
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.markdown(f"""
                                <div class="evidence-box">
                                    <div class="evidence-title">Evidence Source {i}</div>
                                    <div class="evidence-content">{doc['content']}</div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No relevant evidence found")

                # Right Column - Clinical Reasoning
                with col_reasoning:
                    st.markdown('<div class="result-column-header">🧠 AI Clinical Reasoning</div>', unsafe_allow_html=True)

                    answer = result["generated_answer"]

                    if answer:
                        # Format answer for better display
                        # Split answer into sentences
                        sentences = answer.split('. ')

                        # First sentence is the main answer (highlighted)
                        if len(sentences) > 0:
                            main_answer = sentences[0].strip()
                            if not main_answer.endswith('.'):
                                main_answer += '.'

                            # Rest is the explanation
                            explanation = '. '.join(sentences[1:]).strip() if len(sentences) > 1 else ''

                            formatted_answer = f"""
                                <div style="font-size: 1.15rem; font-weight: 700; color: #10b981 !important; margin-bottom: 1rem; line-height: 1.6;">
                                    {main_answer}
                                </div>
                                <div style="font-size: 0.95rem; color: #e0e0e0 !important; line-height: 1.7;">
                                    {explanation}
                                </div>
                            """
                        else:
                            formatted_answer = f'<div style="color: #ffffff !important;">{answer}</div>'

                        st.markdown(f"""
                            <div class="reasoning-box">
                                <div class="reasoning-header" style="font-size: 1.1rem; font-weight: 700; color: #10b981 !important; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #10b981;">
                                    📋 Clinical Answer
                                </div>
                                <div class="reasoning-content">{formatted_answer}</div>
                                <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #2d5a44;">
                                    <div style="font-size: 0.85rem; color: #a0a0a0 !important; font-style: italic;">
                                        💡 Evidence-based answer from RAG + Llama-3-8B
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No response generated")

                st.markdown('</div>', unsafe_allow_html=True)

                # Additional Options
                col1, col2 = st.columns([3, 1])

                with col1:
                    with st.expander("🔍 View Full Technical Details"):
                        st.code(result["prompt"], language="text")

                with col2:
                    # Download
                    download_text = f"""
RACE Clinical Analysis Report
=============================

QUERY:
{query}

RETRIEVED EVIDENCE:
"""
                    for i, doc in enumerate(retrieved_docs, 1):
                        download_text += f"\n[Evidence {i}]\n{doc['content']}\n"

                    download_text += f"""
AI CLINICAL REASONING:
{answer}

Generated by RACE - RAG-Optimized Clinical Reasoning Engine
"""
                    st.download_button(
                        label="📥 Download",
                        data=download_text,
                        file_name="race_clinical_analysis.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                with st.expander("🐛 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #ffffff; padding: 2rem 0;">
            <p style="color: #ffffff !important;"><strong>RACE v1.0</strong> | RAG-Optimized Clinical Reasoning Engine</p>
            <p style="color: #ffffff !important;">Built with ❤️ using Streamlit, LangChain & Llama-3 | For Educational Purposes Only</p>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
