"""
Clinical Reasoning Engine for RACE
Combines RAG retrieval with fine-tuned Llama-3 for medical question answering
"""

import os
import torch
from typing import List, Dict, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
ADAPTER_PATH = "models/race_adapter"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "clinical_guidelines"

# Generation Parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

# Environment
HF_TOKEN = os.getenv("HF_TOKEN")


# =============================================================================
# CLINICAL REASONING ENGINE
# =============================================================================

class ClinicalReasoningEngine:
    """
    RAG-powered clinical reasoning engine combining vector retrieval with
    fine-tuned Llama-3 for evidence-based medical question answering.
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        adapter_path: str = ADAPTER_PATH,
        chroma_db_dir: str = CHROMA_DB_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        load_in_4bit: bool = True,
        device: str = "auto"
    ):
        """
        Initialize the Clinical Reasoning Engine.

        Args:
            base_model: Base model identifier (Llama-3-8B)
            adapter_path: Path to fine-tuned LoRA adapter
            chroma_db_dir: Path to ChromaDB vector store
            embedding_model: Sentence transformer model for embeddings
            load_in_4bit: Whether to use 4-bit quantization
            device: Device to load model on ('auto', 'cuda', 'cpu')
        """
        print("=" * 70)
        print("Initializing Clinical Reasoning Engine")
        print("=" * 70)

        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.device = device
        self.load_in_4bit = load_in_4bit

        # Initialize components
        self._load_vector_store(chroma_db_dir, embedding_model)
        self._load_model_and_tokenizer()

        print("\n" + "=" * 70)
        print("[OK] Clinical Reasoning Engine Ready")
        print("=" * 70)

    def _load_vector_store(self, chroma_db_dir: str, embedding_model: str):
        """Load ChromaDB vector store for retrieval."""
        print(f"\n[1/2] Loading Vector Store")
        print(f"  Database: {chroma_db_dir}")
        print(f"  Embeddings: {embedding_model}")

        if not os.path.exists(chroma_db_dir):
            raise FileNotFoundError(
                f"ChromaDB directory not found: {chroma_db_dir}\n"
                f"Please run 'python src/ingest.py' first to create the vector database."
            )

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load persistent ChromaDB
        self.vectorstore = Chroma(
            persist_directory=chroma_db_dir,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )

        # Verify database has content
        collection_count = self.vectorstore._collection.count()
        print(f"[OK] Vector store loaded ({collection_count} documents)")

        if collection_count == 0:
            print("[WARNING] Vector store is empty!")

    def _load_model_and_tokenizer(self):
        """Load base model with adapter and tokenizer."""
        print(f"\n[2/2] Loading Model")
        print(f"  Base: {self.base_model_name}")
        print(f"  Adapter: {self.adapter_path}")

        # Check if adapter exists
        adapter_exists = os.path.exists(self.adapter_path)
        if not adapter_exists:
            print(f"[WARNING] Adapter not found at {self.adapter_path}")
            print("  Loading base model without fine-tuning...")

        # Configure quantization
        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            token=HF_TOKEN,
            trust_remote_code=True,
        )

        # Load adapter if it exists
        if adapter_exists:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                is_trainable=False
            )
            print(f"[OK] Model loaded with fine-tuned adapter")
        else:
            print(f"[OK] Base model loaded (no adapter)")

        # Set to evaluation mode
        self.model.eval()

        memory_footprint = self.model.get_memory_footprint() / 1e9
        print(f"  Memory footprint: {memory_footprint:.2f} GB")

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, str]]:
        """
        Retrieve relevant context from the vector database.

        Args:
            query: User's medical question
            top_k: Number of relevant chunks to retrieve

        Returns:
            List of dictionaries containing retrieved documents
        """
        # Perform similarity search
        results = self.vectorstore.similarity_search(query, k=top_k)

        # Format results
        retrieved_docs = []
        for i, doc in enumerate(results):
            retrieved_docs.append({
                "chunk_id": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })

        return retrieved_docs

    def _format_context(self, retrieved_docs: List[Dict[str, str]]) -> str:
        """Format retrieved documents into a context string."""
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(doc["content"])

        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the prompt for the language model.

        Format:
        ### Context: {retrieved context}
        ### Question: {user question}
        ### Answer:
        """
        prompt = f"""### Context: {context}
### Question: {query}
### Answer:"""

        return prompt

    def generate_response(
        self,
        query: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        return_full_text: bool = False
    ) -> Dict[str, any]:
        """
        Generate a clinical reasoning response using RAG + fine-tuned model.

        Args:
            query: Medical question from user
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            return_full_text: Whether to return full prompt + response

        Returns:
            Dictionary containing:
                - query: Original question
                - retrieved_context: Evidence from vector DB
                - generated_answer: Model's response
                - full_response: Complete model output (if return_full_text=True)
        """
        print(f"\n{'='*70}")
        print("Processing Query")
        print(f"{'='*70}")
        print(f"Query: {query}\n")

        # Step 1: Retrieve relevant context
        print("[Step 1/2] Retrieving relevant context from knowledge base...")
        retrieved_docs = self.retrieve(query, top_k=2)

        print(f"[OK] Retrieved {len(retrieved_docs)} relevant chunks:")
        for i, doc in enumerate(retrieved_docs, 1):
            preview = doc["content"][:100].replace('\n', ' ')
            print(f"  [{i}] {preview}...")

        # Step 2: Format context and build prompt
        context = self._format_context(retrieved_docs)
        prompt = self._build_prompt(query, context)

        # Step 3: Generate response
        print(f"\n[Step 2/2] Generating response from fine-tuned model...")

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated answer (remove prompt)
            if not return_full_text and "### Answer:" in full_response:
                generated_answer = full_response.split("### Answer:")[-1].strip()
            else:
                # If no ### Answer: marker found, take everything after the prompt
                if prompt in full_response:
                    generated_answer = full_response.replace(prompt, "").strip()
                else:
                    generated_answer = full_response

            # Fallback if generation is empty or just repeats prompt
            if not generated_answer or len(generated_answer) < 20:
                # Use first retrieved document as fallback
                generated_answer = f"Based on the retrieved medical guidelines:\n\n{context[:500]}..."

            print(f"[OK] Response generated ({len(generated_answer)} characters)")

        except Exception as e:
            print(f"[WARNING] Model generation failed: {e}")
            print("[OK] Using fallback: returning retrieved context")
            # Fallback: provide context-based answer
            generated_answer = f"Based on the retrieved medical guidelines:\n\n{context}"
            full_response = prompt + "\n" + generated_answer

        # Prepare result
        result = {
            "query": query,
            "retrieved_context": retrieved_docs,
            "generated_answer": generated_answer,
            "prompt": prompt,
        }

        if return_full_text:
            result["full_response"] = full_response

        return result

    def chat(self, query: str, verbose: bool = True) -> str:
        """
        Simplified chat interface that returns only the answer.

        Args:
            query: Medical question
            verbose: Whether to print retrieval and generation info

        Returns:
            Generated answer string
        """
        result = self.generate_response(query)

        if verbose:
            print(f"\n{'='*70}")
            print("Retrieved Evidence:")
            print(f"{'='*70}")
            for i, doc in enumerate(result["retrieved_context"], 1):
                print(f"\n[Chunk {i}]")
                print(doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"])

            print(f"\n{'='*70}")
            print("Clinical Reasoning Answer:")
            print(f"{'='*70}")
            print(result["generated_answer"])
            print(f"{'='*70}\n")

        return result["generated_answer"]


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Interactive CLI for testing the inference engine."""
    print("\n" + "=" * 70)
    print("RACE: Clinical Reasoning Engine - Interactive Mode")
    print("=" * 70)

    # Initialize engine
    try:
        engine = ClinicalReasoningEngine()
    except Exception as e:
        print(f"\n❌ Error initializing engine: {e}")
        return 1

    # Interactive loop
    print("\nType your medical question (or 'quit' to exit)")
    print("-" * 70)

    while True:
        try:
            query = input("\n🏥 Question: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting... Goodbye! 👋")
                break

            # Generate response
            answer = engine.chat(query, verbose=True)

        except KeyboardInterrupt:
            print("\n\nExiting... Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    exit(main())