"""
Interactive Model Testing Script
Test your fine-tuned model with custom prompts and see the results!
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your fine-tuned model (change if needed)
MODEL_PATH = "./output/final_model"

# Generation parameters - adjust these to control output style
MAX_LENGTH = 150              # Maximum length of generated text
TEMPERATURE = 0.7             # 0.0 = deterministic, 1.0 = very random
TOP_P = 0.9                   # Nucleus sampling (0.9 = use top 90% probability mass)
TOP_K = 50                    # Consider only top K tokens (0 = disabled)
REPETITION_PENALTY = 1.1      # Penalize repeated tokens (1.0 = no penalty)
DO_SAMPLE = True              # Enable sampling for more creative outputs

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================

print("=" * 70)
print("🤖 FINE-TUNED MODEL TESTER")
print("=" * 70)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ ERROR: Model not found at '{MODEL_PATH}'")
    print("\n💡 Have you run simple_finetune.py yet?")
    print("   That script trains and saves the model to ./output/final_model")
    sys.exit(1)

# Check GPU
if torch.cuda.is_available():
    print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("\n⚠️  No GPU detected - using CPU (will be slower)")
    device = "cpu"

# Load tokenizer
print(f"\nLoading model from: {MODEL_PATH}")
print("Please wait...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

# Get model size
num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded successfully ({num_params/1e6:.0f}M parameters)")

# ============================================================================
# HELPER FUNCTION FOR GENERATION
# ============================================================================

def generate_text(prompt, show_settings=False):
    """
    Generate text from the model given a prompt.
    
    Args:
        prompt (str): The input text to start generation from
        show_settings (bool): Whether to display generation settings
    
    Returns:
        str: The generated text
    """
    # Show generation settings if requested
    if show_settings:
        print("\n⚙️  Generation Settings:")
        print(f"   • Max Length: {MAX_LENGTH}")
        print(f"   • Temperature: {TEMPERATURE} (higher = more random)")
        print(f"   • Top-p: {TOP_P} (nucleus sampling)")
        print(f"   • Top-k: {TOP_K}")
        print(f"   • Repetition Penalty: {REPETITION_PENALTY}")
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    print("\n🔄 Generating...")
    with torch.no_grad():  # Don't track gradients (saves memory)
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """
    Run the interactive prompt loop where users can test multiple prompts.
    """
    print("\n" + "=" * 70)
    print("🎯 INTERACTIVE MODE")
    print("=" * 70)
    print("\nType your prompts and see what the model generates!")
    print("Commands:")
    print("  • Type any text to generate")
    print("  • 'settings' - show generation settings")
    print("  • 'quit' or 'exit' - exit the program")
    print("=" * 70)
    
    # Show settings on first run
    generate_text("", show_settings=True)
    
    while True:
        print("\n" + "-" * 70)
        
        # Get user input
        try:
            prompt = input("\n💬 Enter prompt (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        # Handle special commands
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if prompt.lower() == 'settings':
            generate_text("", show_settings=True)
            continue
        
        if not prompt:
            print("⚠️  Please enter a prompt!")
            continue
        
        # Generate and display output
        print("\n📝 Prompt:")
        print(f"   {prompt}")
        
        output = generate_text(prompt)
        
        print("\n🤖 Generated Text:")
        print("-" * 70)
        print(output)
        print("-" * 70)

# ============================================================================
# SINGLE PROMPT MODE
# ============================================================================

def single_prompt_mode(prompt):
    """
    Generate text for a single prompt (useful for command-line usage).
    
    Args:
        prompt (str): The prompt to generate from
    """
    print(f"\n📝 Prompt: {prompt}")
    output = generate_text(prompt, show_settings=True)
    print("\n🤖 Generated Text:")
    print("=" * 70)
    print(output)
    print("=" * 70)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if a prompt was provided as command-line argument
    if len(sys.argv) > 1:
        # Single prompt mode: python test_model.py "Your prompt here"
        prompt = " ".join(sys.argv[1:])
        single_prompt_mode(prompt)
    else:
        # Interactive mode: python test_model.py
        interactive_mode()

