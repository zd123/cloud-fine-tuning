"""
Simple Fine-Tuning Blueprint
A minimal working example for fine-tuning a language model on custom text.
Perfect for testing your Lambda Labs setup.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import glob
import os

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Small, fast model
DATA_DIR = "./training_data"                # Folder with your .txt files
OUTPUT_DIR = "./output"                     # Where to save the model
MAX_LENGTH = 512                            # Sequence length
EPOCHS = 3                                  # Number of training passes

# ============================================================================
# STEP 1: Load and prepare data
# ============================================================================

print("=" * 60)
print("STEP 1: Loading training data...")
print("=" * 60)

# Find all .txt files
txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
if not txt_files:
    raise ValueError(f"No .txt files found in {DATA_DIR}. Please add training data!")

# Read and combine all text files
all_text = []
for file in txt_files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        all_text.append(text)
        print(f"  ✓ Loaded: {file} ({len(text):,} chars)")

combined_text = "\n\n".join(all_text)
print(f"\nTotal text: {len(combined_text):,} characters\n")

# ============================================================================
# STEP 2: Load model and tokenizer
# ============================================================================

print("=" * 60)
print("STEP 2: Loading model and tokenizer...")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ No GPU! Training will be SLOW.")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

# Load model
# Using bfloat16 for better numerical stability with modern GPUs
# Note: Some GPUs (A10, A100, H100) support bf16, older ones will fall back to fp16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded ({num_params/1e6:.0f}M parameters)\n")

# ============================================================================
# STEP 3: Tokenize and create dataset
# ============================================================================

print("=" * 60)
print("STEP 3: Creating dataset...")
print("=" * 60)

# Tokenize entire text
tokens = tokenizer(combined_text, return_tensors="pt", truncation=False)
token_ids = tokens['input_ids'][0].tolist()
print(f"Total tokens: {len(token_ids):,}")

# Split into chunks
chunks = []
for i in range(0, len(token_ids) - MAX_LENGTH, MAX_LENGTH):
    chunk = token_ids[i:i + MAX_LENGTH]
    chunks.append({'input_ids': chunk})

print(f"Created {len(chunks)} training chunks")

# Create dataset and split
dataset = Dataset.from_list(chunks)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']
print(f"Train: {len(train_dataset)} | Validation: {len(eval_dataset)}\n")

# ============================================================================
# STEP 4: Configure training
# ============================================================================

print("=" * 60)
print("STEP 4: Configuring training...")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    # Use bf16 instead of fp16 for better stability with device_map="auto"
    # bf16 doesn't require gradient scaling and works better with mixed precision
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=False,  # Explicitly disable fp16 to avoid gradient scaling conflicts
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    warmup_steps=50,
    seed=42,
)

print(f"✓ Training for {EPOCHS} epochs")
print(f"✓ Batch size: 2 (effective: 8 with accumulation)")
print(f"✓ Learning rate: {training_args.learning_rate}\n")

# ============================================================================
# STEP 5: Train!
# ============================================================================

print("=" * 60)
print("STEP 5: Starting training...")
print("=" * 60)

# Create data collator - this prepares batches for training
# mlm=False means causal language modeling (predict next token)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize the trainer - this handles all the training logic
# Using processing_class instead of tokenizer (updated API)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# START TRAINING
# This will show a progress bar and training metrics
trainer.train()

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)

# ============================================================================
# STEP 6: Save the model
# ============================================================================

print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR + "/final_model")
tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")
print(f"✓ Model saved to {OUTPUT_DIR}/final_model")

# ============================================================================
# STEP 7: Compare original vs fine-tuned model
# ============================================================================

print("\n" + "=" * 60)
print("STEP 7: Comparing models (before vs after fine-tuning)...")
print("=" * 60)

# Load the original model for comparison
print("\nLoading original model for comparison...")
original_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
print("✓ Original model loaded\n")

# Test prompts - you can customize these
test_prompts = [
    "Once upon a time",
    "The meaning of life is",
    "In my opinion,",
]

# Generate from both models for comparison
for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*60}")
    print(f"TEST {i}: '{prompt}'")
    print(f"{'='*60}\n")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with ORIGINAL model
    print("🔵 ORIGINAL MODEL OUTPUT:")
    print("-" * 60)
    with torch.no_grad():
        original_outputs = original_model.generate(
            **inputs, 
            max_length=100,           # Maximum length of generation
            temperature=0.7,          # Controls randomness (0.0 = deterministic, 1.0 = very random)
            do_sample=True,           # Enable sampling (more creative)
            top_p=0.9,                # Nucleus sampling (consider top 90% probability mass)
            repetition_penalty=1.1    # Penalize repetition
        )
    original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
    print(original_text)
    
    # Generate with FINE-TUNED model
    print("\n🟢 FINE-TUNED MODEL OUTPUT:")
    print("-" * 60)
    with torch.no_grad():
        finetuned_outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    finetuned_text = tokenizer.decode(finetuned_outputs[0], skip_special_tokens=True)
    print(finetuned_text)
    print()

print("=" * 60)
print("✓ TRAINING COMPLETE! Your fine-tuned model is ready.")
print(f"✓ Model saved to: {OUTPUT_DIR}/final_model")
print("=" * 60)
print("\n💡 TIP: Use test_model.py to test with custom prompts!")

