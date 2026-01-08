# RACE: RAG-Optimized Clinical Reasoning Engine

A production-grade compound AI system that combines retrieval-augmented generation (RAG) with QLoRA-fine-tuned Llama-3-8B to deliver evidence-based medical reasoning with hallucination mitigation.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 The Problem

Modern large language models face critical limitations in medical applications:

1. **Hallucination Risk**: Generic LLMs generate plausible but factually incorrect medical information without grounding in verified sources
2. **Lack of Evidence**: Standard models cannot cite specific medical guidelines or research to support recommendations
3. **Insufficient Reasoning**: Out-of-the-box models lack domain-specific chain-of-thought reasoning capabilities for complex clinical scenarios
4. **Resource Constraints**: Medical-grade models are typically deployed on expensive cloud infrastructure, limiting accessibility

---

## 💡 The Solution

**RACE** implements a compound AI architecture that addresses these challenges through:

### 1. **Retrieval-Augmented Generation (RAG)**
- Vector database (ChromaDB) ingests medical knowledge bases and clinical guidelines
- Semantic search retrieves relevant evidence chunks before generation
- Grounds model outputs in verifiable medical literature

### 2. **Domain-Specific Fine-Tuning**
- Llama-3-8B fine-tuned on `OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B` dataset
- QLoRA (Quantized Low-Rank Adaptation) enables parameter-efficient training
- Optimized for chain-of-thought (CoT) reasoning in clinical contexts

### 3. **Memory-Efficient Deployment**
- 4-bit quantization via `bitsandbytes` reduces memory footprint by ~75%
- Runs on consumer GPUs (8GB+ VRAM) while maintaining reasoning quality
- Gradient checkpointing and paged optimizers prevent OOM errors

### System Architecture

```
User Query
    ↓
┌───────────────────────────────────────┐
│   RAG Retrieval Layer (ChromaDB)      │
│   • Semantic search (all-MiniLM-L6)   │
│   • Top-K evidence retrieval          │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│   Fine-Tuned LLM (Llama-3-8B)         │
│   • QLoRA adapters (r=16)             │
│   • 4-bit NF4 quantization            │
│   • Chain-of-thought reasoning        │
└───────────────────────────────────────┘
    ↓
Evidence + Reasoned Answer
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Base Model** | Meta Llama-3-8B | Foundation language model |
| **Fine-Tuning** | QLoRA (PEFT) | Parameter-efficient adaptation |
| **Quantization** | bitsandbytes (4-bit NF4) | Memory optimization |
| **Vector DB** | ChromaDB | Persistent embeddings storage |
| **Embeddings** | sentence-transformers | Semantic text encoding |
| **RAG Framework** | LangChain | Retrieval orchestration |
| **Training** | TRL (SFTTrainer) | Supervised fine-tuning pipeline |
| **Interface** | Streamlit | Interactive web UI |
| **Compute** | PyTorch 2.1+ | GPU acceleration |

---

## ⚡ Performance & Optimization

### Memory Efficiency
- **4-bit quantization** reduces model size from ~32GB to ~5.5GB
- **LoRA adapters** train only 0.1% of parameters (~8M trainable vs ~8B total)
- **Gradient checkpointing** trades compute for memory during training

### Hardware Requirements
| Configuration | VRAM | Use Case |
|--------------|------|----------|
| Minimum | 8GB | Inference only (RTX 3050, T4) |
| Recommended | 12GB+ | Inference + fine-tuning (RTX 3060, T4) |
| Optimal | 16GB+ | Full training pipeline (A100, V100) |

### Inference Speed
- **First query**: ~30-60s (model loading)
- **Subsequent queries**: ~5-10s (cached model)
- **Retrieval latency**: <1s for top-K search

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
python --version

# GPU with CUDA support (optional but recommended)
nvidia-smi
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sathishsravanakumar/RACE.git
cd RACE
```

2. **Set up virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Configure environment**
```bash
# Set your HuggingFace token for Llama-3 access
set HF_TOKEN=your_huggingface_token_here  # Windows
# export HF_TOKEN=your_token  # Linux/Mac
```

### Usage Pipeline

#### 1. Ingest Medical Knowledge
```bash
python src/ingest.py
```
- Loads clinical guidelines from `data/clinical_guidelines.txt`
- Creates embeddings using sentence-transformers
- Stores in persistent ChromaDB (`chroma_db/`)

#### 2. Fine-Tune Model (Optional)
```bash
python src/train.py
```
- Downloads Llama-3-8B and applies 4-bit quantization
- Fine-tunes using QLoRA on medical reasoning dataset
- Saves adapter to `models/race_adapter/`
- **Note**: Requires ~200 examples for demo; increase `NUM_SAMPLES` for production

#### 3. Launch Web Interface
```bash
streamlit run app.py
```
- Opens browser at `http://localhost:8501`
- Interactive UI with evidence retrieval + reasoning display

#### 4. Programmatic Usage
```python
from src.inference import ClinicalReasoningEngine

# Initialize engine
engine = ClinicalReasoningEngine()

# Query with automatic RAG retrieval
result = engine.generate_response(
    "What is the recommended starting dose of metformin?"
)

print("Evidence:", result["retrieved_context"])
print("Answer:", result["generated_answer"])
```

---

## 📁 Project Structure

```
RACE/
├── data/
│   └── clinical_guidelines.txt      # Medical knowledge base
├── models/
│   └── race_adapter/                # Fine-tuned LoRA weights
├── src/
│   ├── ingest.py                    # RAG ingestion pipeline
│   ├── train.py                     # QLoRA fine-tuning script
│   └── inference.py                 # Reasoning engine + API
├── chroma_db/                       # Persistent vector database
├── app.py                           # Streamlit web interface
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🎓 Technical Highlights

### 1. **Advanced Fine-Tuning Techniques**
- **QLoRA Configuration**:
  - Rank `r=16`, Alpha `16` for optimal capacity/efficiency trade-off
  - Target modules: `q_proj`, `v_proj` (attention layers)
  - Double quantization for nested memory savings
- **Training Optimization**:
  - Batch size `1` with gradient accumulation `4`
  - `paged_adamw_8bit` optimizer for memory efficiency
  - Cosine learning rate schedule with warmup

### 2. **RAG Implementation**
- **Text Splitting**: `RecursiveCharacterTextSplitter` with 500-char chunks, 50-char overlap
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dim, fast inference)
- **Retrieval**: Top-2 similarity search with normalized embeddings

### 3. **Prompt Engineering**
```
### Context: {retrieved_medical_evidence}
### Question: {user_clinical_query}
### Answer:
```
- Structured format guides model to leverage evidence
- Mirrors training data format for consistency

### 4. **Production-Ready Design**
- `@st.cache_resource` decorator prevents redundant model loads
- Error handling with graceful degradation
- Modular architecture (RAG, training, inference separated)
- Configuration via environment variables

---

## 📊 Example Use Cases

1. **Clinical Decision Support**: "What are contraindications for metformin in CKD patients?"
2. **Medical Education**: "Explain the mechanism and dosing of GLP-1 receptor agonists"
3. **Treatment Planning**: "When should basal insulin be initiated in T2DM?"
4. **Evidence Retrieval**: Automatically cite relevant guidelines for clinical questions

---

## 🔬 Future Enhancements

- [ ] Multi-modal support (integrate medical imaging, lab results)
- [ ] Expand knowledge base to broader medical specialties
- [ ] Implement RLHF (Reinforcement Learning from Human Feedback)
- [ ] Add evaluation metrics (ROUGE, BERTScore, medical accuracy benchmarks)
- [ ] Deploy as REST API with FastAPI
- [ ] Integrate with FHIR standards for EHR compatibility

---

## 🤝 Contributing

This is a portfolio project demonstrating ML Engineering capabilities. For collaboration or inquiries:

- **Issues**: Report bugs or suggest features via GitHub Issues
- **Pull Requests**: Contributions welcome (please open issue first)
- **Contact**: sathishsravanakumar@gmail.com

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Meta AI**: Llama-3-8B foundation model
- **OpenMed**: Medical reasoning dataset
- **Hugging Face**: Transformers ecosystem and PEFT library
- **bitsandbytes**: Quantization techniques (Tim Dettmers et al.)

---

## 📈 Project Metrics

- **Model Size**: 5.5GB (4-bit) vs 32GB (FP32)
- **Trainable Parameters**: 8.4M (0.1% of total 8B)
- **Training Time**: ~15-20 min on T4 GPU (200 samples)
- **Inference Latency**: 5-10s per query (GPU), 30-60s (CPU)

---

**Built with ❤️ to demonstrate advanced ML Engineering practices in healthcare AI**

*For educational and portfolio purposes. Not intended for clinical use without proper validation and regulatory approval.*
