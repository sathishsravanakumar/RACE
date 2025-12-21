"""
RACE: RAG-Optimized Clinical Reasoning Engine
Streamlit Web Interface
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
    initial_sidebar_state="expanded"
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
        with st.spinner("🔄 Loading Clinical Reasoning Engine... This may take a minute..."):
            engine = ClinicalReasoningEngine()
        return engine, None
    except Exception as e:
        return None, str(e)


# =============================================================================
# CUSTOM CSS
# =============================================================================

def load_custom_css():
    """Apply custom styling to the app."""
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stTextArea textarea {
            font-size: 1.1rem;
        }
        .evidence-box {
            background-color: #e7f3ff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1f77b4;
            margin-bottom: 1rem;
        }
        .reasoning-box {
            background-color: #e8f5e9;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 5px solid #4caf50;
        }
        .chunk-header {
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .footer {
            text-align: center;
            color: #999;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #eee;
        }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application logic."""

    # Apply custom CSS
    load_custom_css()

    # Header
    st.markdown('<div class="main-header">🏥 RACE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">RAG-Optimized Clinical Reasoning Engine</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About RACE")
        st.markdown("""
        **RACE** combines:
        - 📚 **RAG (Retrieval-Augmented Generation)**: Retrieves relevant medical evidence from a knowledge base
        - 🧠 **Fine-Tuned LLM**: Llama-3-8B fine-tuned on medical reasoning data using QLoRA

        **How it works:**
        1. Enter a clinical question or scenario
        2. System retrieves relevant evidence from medical guidelines
        3. Fine-tuned model generates a reasoned response
        4. View both evidence and reasoning side-by-side
        """)

        st.markdown("---")

        st.header("⚙️ Settings")

        # Generation parameters
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=512,
            value=256,
            step=64,
            help="Maximum length of generated response"
        )

        top_k_retrieval = st.slider(
            "Retrieval Chunks",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Number of evidence chunks to retrieve"
        )

        st.markdown("---")

        st.header("📊 System Status")

    # Load model
    engine, error = load_reasoning_engine()

    # Display status in sidebar
    with st.sidebar:
        if engine:
            st.success("✅ Model Loaded")
            st.info(f"📍 Vector DB: {engine.vectorstore._collection.count()} documents")
        else:
            st.error("❌ Model Failed to Load")
            if error:
                st.error(f"Error: {error}")

    # Main content
    if not engine:
        st.error("⚠️ Failed to initialize Clinical Reasoning Engine")
        st.error(f"**Error Details:** {error}")

        st.markdown("### Troubleshooting Steps:")
        st.markdown("""
        1. **Run the ingestion script first:**
           ```bash
           python src/ingest.py
           ```

        2. **Ensure you have the fine-tuned adapter** (optional):
           ```bash
           python src/train.py
           ```

        3. **Check that all dependencies are installed:**
           ```bash
           pip install -r requirements.txt
           ```

        4. **Verify your HuggingFace token is set** (for Llama-3):
           ```bash
           set HF_TOKEN=your_token_here
           ```
        """)
        return

    # Input Section
    st.markdown("### 📝 Clinical Scenario")

    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is the recommended starting dose of metformin?
        - What are the contraindications for metformin therapy?
        - When should insulin therapy be initiated in type 2 diabetes?
        - What is the role of GLP-1 receptor agonists in diabetes management?
        - How should metformin dosage be adjusted in patients with renal impairment?
        - What are the first-line treatment options for type 2 diabetes?
        """)

    # Text input
    query = st.text_area(
        "Enter Clinical Scenario:",
        height=150,
        placeholder="Example: A 55-year-old patient with newly diagnosed type 2 diabetes (HbA1c 8.5%) and normal renal function. What is the recommended first-line treatment?",
        label_visibility="collapsed"
    )

    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

    # Process query
    if analyze_button:
        if not query.strip():
            st.warning("⚠️ Please enter a clinical scenario or question.")
            return

        # Show processing status
        with st.spinner("🔄 Analyzing clinical scenario..."):
            try:
                # Generate response
                result = engine.generate_response(
                    query=query,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k_retrieval
                )

                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")

                # Two-column layout
                col_evidence, col_reasoning = st.columns(2)

                # Column 1: Retrieved Evidence
                with col_evidence:
                    st.markdown("### 📚 Retrieved Evidence")
                    st.markdown("*Relevant information from medical knowledge base:*")

                    retrieved_docs = result["retrieved_context"]

                    if retrieved_docs:
                        for i, doc in enumerate(retrieved_docs, 1):
                            with st.container():
                                st.info(f"""
**Evidence Chunk {i}:**

{doc['content']}
                                """)
                    else:
                        st.warning("No relevant evidence found in knowledge base.")

                # Column 2: Generated Reasoning
                with col_reasoning:
                    st.markdown("### 🧠 Clinical Reasoning")
                    st.markdown("*AI-generated analysis based on evidence:*")

                    answer = result["generated_answer"]

                    if answer:
                        st.success(answer)
                    else:
                        st.warning("No response generated.")

                # Additional info (collapsible)
                with st.expander("🔍 View Full Prompt"):
                    st.code(result["prompt"], language="text")

                # Download option
                st.markdown("---")

                # Format response for download
                download_text = f"""
RACE Clinical Reasoning Analysis
================================

QUERY:
{query}

RETRIEVED EVIDENCE:
-------------------
"""
                for i, doc in enumerate(retrieved_docs, 1):
                    download_text += f"\nEvidence Chunk {i}:\n{doc['content']}\n"

                download_text += f"""
CLINICAL REASONING:
-------------------
{answer}

Generated by RACE - RAG-Optimized Clinical Reasoning Engine
"""

                st.download_button(
                    label="📥 Download Analysis",
                    data=download_text,
                    file_name="race_analysis.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                import traceback
                with st.expander("🐛 Error Details"):
                    st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">Built with ❤️ using Streamlit, LangChain, and Llama-3 | '
        'RACE v1.0 | For educational purposes only</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
