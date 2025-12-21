"""
QLoRA Fine-Tuning Script for RACE
Fine-tunes Llama-3-8B on medical reasoning data using 4-bit quantization
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model Configuration
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR = "models/race_adapter"
DATASET_NAME = "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B"

# Training Configuration
NUM_SAMPLES = None  # None = use full dataset, or set to specific number for testing
MAX_SEQ_LENGTH = 512
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 3  # Increased for better convergence on full dataset
LEARNING_RATE = 2e-4
SAVE_STEPS = 500  # Save less frequently with larger dataset
LOGGING_STEPS = 50
MAX_STEPS = -1  # -1 = train for full epochs, or set specific step limit

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# Environment
HF_TOKEN = os.getenv("HF_TOKEN")

# =============================================================================
# QUANTIZATION CONFIG (4-bit for memory efficiency)
# =============================================================================

def get_bnb_config():
    """Configure 4-bit quantization using BitsAndBytes."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Nested quantization for better memory
        bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
        bnb_4bit_compute_dtype=torch.float16,  # Computation dtype
    )
    return bnb_config


# =============================================================================
# LORA CONFIG
# =============================================================================

def get_lora_config():
    """Configure LoRA parameters."""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_dataset():
    """Load dataset and format it for training."""
    print(f"\n[1/5] Loading dataset: {DATASET_NAME}")

    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="train")

    total_available = len(dataset)
    print(f"Total available examples: {total_available:,}")

    # Slice to first NUM_SAMPLES examples if specified
    if NUM_SAMPLES is not None:
        dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
        print(f"[OK] Using {len(dataset):,} examples (subset for testing)")
    else:
        print(f"[OK] Using full dataset: {len(dataset):,} examples")

    print(f"\nSample data structure:")
    print(dataset[0])

    return dataset


def format_prompt(example):
    """
    Format the dataset into a structured prompt.
    Expected fields: 'question' and 'reasoning' (adjust based on actual dataset schema)
    """
    # Adjust field names based on actual dataset structure
    # Common alternatives: 'instruction', 'input', 'output', 'response', 'answer'

    question = example.get('question', example.get('instruction', example.get('input', '')))
    reasoning = example.get('reasoning', example.get('output', example.get('response', '')))

    prompt = f"""### Question: {question}
### Reasoning: {reasoning}"""

    return {"text": prompt}


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_tokenizer():
    """Load base model with 4-bit quantization and tokenizer."""
    print(f"\n[2/5] Loading model: {BASE_MODEL}")

    # Check for HF token
    if not HF_TOKEN:
        print("⚠ WARNING: HF_TOKEN not found in environment variables")
        print("You may need to authenticate for gated models")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    print(f"✓ Model loaded in 4-bit precision")
    print(f"✓ Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapters to the model."""
    print(f"\n[3/5] Applying LoRA configuration")

    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✓ LoRA applied successfully")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")

    return model


# =============================================================================
# TRAINING
# =============================================================================

def get_training_args():
    """Configure training arguments."""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,  # -1 for full epochs, or set specific limit
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,  # Mixed precision training
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=3,  # Keep more checkpoints for full dataset training
        warmup_steps=100,  # More warmup for larger dataset
        warmup_ratio=0.03,  # 3% of training as warmup
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        report_to="none",  # Disable wandb/tensorboard for now
        gradient_checkpointing=True,  # Save memory
        logging_first_step=True,
        eval_strategy="no",  # No evaluation during training
        save_strategy="steps",
    )
    return training_args


def train_model(model, tokenizer, dataset):
    """Train the model using SFTTrainer."""
    print(f"\n[4/5] Starting training")

    # Format dataset
    formatted_dataset = dataset.map(
        format_prompt,
        remove_columns=dataset.column_names,
        desc="Formatting prompts"
    )

    training_args = get_training_args()

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,  # Don't pack sequences
    )

    # Train
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    return trainer


# =============================================================================
# SAVE MODEL
# =============================================================================

def save_adapter(trainer):
    """Save the fine-tuned LoRA adapter."""
    print(f"\n[5/5] Saving adapter to: {OUTPUT_DIR}")

    # Save adapter
    trainer.model.save_pretrained(OUTPUT_DIR)
    trainer.tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✓ Adapter saved successfully")
    print(f"  Location: {os.path.abspath(OUTPUT_DIR)}")

    # Save training configuration
    config_path = os.path.join(OUTPUT_DIR, "training_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Base Model: {BASE_MODEL}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Samples: {NUM_SAMPLES}\n")
        f.write(f"LoRA r: {LORA_R}\n")
        f.write(f"LoRA alpha: {LORA_ALPHA}\n")
        f.write(f"Batch size: {PER_DEVICE_BATCH_SIZE}\n")
        f.write(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")

    print(f"✓ Training config saved to: {config_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("RACE: QLoRA Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"Base Model: {BASE_MODEL}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available. Training will be very slow on CPU!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    try:
        # Load dataset
        dataset = load_and_prepare_dataset()

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()

        # Apply LoRA
        model = apply_lora(model)

        # Train
        trainer = train_model(model, tokenizer, dataset)

        # Save adapter
        save_adapter(trainer)

        print("\n" + "=" * 60)
        print("✓ FINE-TUNING COMPLETE!")
        print("=" * 60)
        print(f"\nThe trained adapter is ready at: {OUTPUT_DIR}")
        print("You can now use this adapter for inference with the base model.")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())