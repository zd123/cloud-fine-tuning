# Testing Your Fine-Tuned Model - Quick Guide

## Overview

After training with `simple_finetune.py`, you have **two ways** to test your fine-tuned model:

1. **Automatic Comparison** - Built into `simple_finetune.py` (runs automatically after training)
2. **Interactive Testing** - Use `test_model.py` for custom prompts

---

## 🔄 Automatic Comparison (Built-in)

`simple_finetune.py` now includes **STEP 7: Compare original vs fine-tuned model**

### What Happens Automatically

After training completes, the script:
1. Loads the original (pre-trained) model
2. Loads your fine-tuned model
3. Tests both with the same prompts
4. Shows side-by-side comparison

### Default Test Prompts

The script tests these prompts by default:
- "Once upon a time"
- "The meaning of life is"
- "In my opinion,"

### Customizing Test Prompts

Edit `simple_finetune.py` around line 204:

```python
test_prompts = [
    "Once upon a time",
    "The meaning of life is",
    "In my opinion,",
    # Add your own prompts here!
    "Your custom prompt",
]
```

### Output Format

```
============================================================
TEST 1: 'Once upon a time'
============================================================

🔵 ORIGINAL MODEL OUTPUT:
------------------------------------------------------------
Once upon a time there was a kingdom...

🟢 FINE-TUNED MODEL OUTPUT:
------------------------------------------------------------
Once upon a time in the Bronx, there was...
```

**Key Insight**: You can see how your fine-tuning changed the model's style!

---

## 🎮 Interactive Testing (`test_model.py`)

For more flexible testing after training completes.

### Usage Method 1: Interactive Mode

Best for testing multiple prompts:

```bash
python test_model.py
```

**Interactive Commands:**
- Type any text → Generate output
- `settings` → Show generation parameters
- `quit` or `exit` → Exit program

**Example Session:**
```
💬 Enter prompt (or 'quit' to exit): Tell me about the Bronx

🔄 Generating...

🤖 Generated Text:
------------------------------------------------------------
Tell me about the Bronx, where I was born and raised...
------------------------------------------------------------

💬 Enter prompt (or 'quit' to exit): What makes music great?

🔄 Generating...

🤖 Generated Text:
------------------------------------------------------------
What makes music great is the energy, the beat, the lyrics...
------------------------------------------------------------

💬 Enter prompt (or 'quit' to exit): quit
👋 Goodbye!
```

### Usage Method 2: Command-Line Mode

Best for quick one-off tests:

```bash
python test_model.py "Your prompt here"
```

**Examples:**
```bash
# Test with a specific prompt
python test_model.py "Once upon a time in New York"

# Multi-word prompts need quotes
python test_model.py "I think the best thing about hip hop is"

# Save output to file
python test_model.py "Write a story" > output.txt
```

---

## ⚙️ Customizing Generation Settings

Edit the top of `test_model.py` to control output style:

```python
# PATH CONFIGURATION
MODEL_PATH = "./output/final_model"  # Where your trained model is saved

# GENERATION SETTINGS
MAX_LENGTH = 150              # How many tokens to generate (longer = more text)
TEMPERATURE = 0.7             # Randomness (0.0 = boring, 1.0 = wild)
TOP_P = 0.9                   # Nucleus sampling (0.9 = top 90% probability)
TOP_K = 50                    # Limit to top K tokens (0 = disabled)
REPETITION_PENALTY = 1.1      # Avoid repeating text (1.0 = no penalty)
DO_SAMPLE = True              # Enable sampling (False = always pick most likely)
```

### Parameter Guide

| Parameter | Low Value | High Value | Use Case |
|-----------|-----------|------------|----------|
| `TEMPERATURE` | 0.1 (safe, predictable) | 1.5 (creative, risky) | Creative writing = high, factual = low |
| `TOP_P` | 0.5 (conservative) | 0.95 (diverse) | More diverse vocabulary = higher |
| `MAX_LENGTH` | 50 (short) | 500 (long) | Depends on your use case |
| `REPETITION_PENALTY` | 1.0 (no penalty) | 2.0 (strong penalty) | If model repeats itself = increase |

### Example Configurations

**Creative Writing Mode:**
```python
MAX_LENGTH = 300
TEMPERATURE = 0.9
TOP_P = 0.95
REPETITION_PENALTY = 1.2
```

**Factual/Predictable Mode:**
```python
MAX_LENGTH = 100
TEMPERATURE = 0.3
TOP_P = 0.8
REPETITION_PENALTY = 1.0
```

**Balanced Mode (Default):**
```python
MAX_LENGTH = 150
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
```

---

## 📊 Comparing Results

### What to Look For

1. **Style Transfer**: Does the fine-tuned model adopt the style of your training data?
2. **Vocabulary**: Does it use words/phrases from your training data?
3. **Topics**: Does it stay on-topic for your domain?
4. **Quality**: Is the output coherent and relevant?

### Example Comparison

**Training Data**: Cardi B lyrics (energetic, conversational, Bronx references)

**Original Model:**
```
"Once upon a time" → generic fairy tale
```

**Fine-Tuned Model:**
```
"Once upon a time" → story with NYC/Bronx flavor, conversational tone
```

---

## 🚀 Advanced Usage

### Testing on Different Hardware

The script automatically detects your hardware:

- **GPU Available**: Uses GPU with bf16/fp16 (fast)
- **CPU Only**: Falls back to CPU (slower but works)

### Loading Custom Model Paths

```python
# Edit test_model.py
MODEL_PATH = "/path/to/your/model"

# Or use environment variable
import os
MODEL_PATH = os.environ.get("MODEL_PATH", "./output/final_model")
```

Then run:
```bash
MODEL_PATH="/custom/path" python test_model.py "test prompt"
```

### Batch Testing Multiple Prompts

Create a file `prompts.txt`:
```
Once upon a time
Tell me about music
What makes art great
```

Run batch test:
```bash
while read prompt; do
    echo "Testing: $prompt"
    python test_model.py "$prompt"
    echo "---"
done < prompts.txt
```

### Using in a Python Script

```python
# Import the model loading code
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("./output/final_model")
tokenizer = AutoTokenizer.from_pretrained("./output/final_model")

# Generate
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

---

## 🐛 Troubleshooting

### Error: "Model not found at './output/final_model'"

**Solution**: You need to run `simple_finetune.py` first to train and save the model.

```bash
python simple_finetune.py  # Train first
python test_model.py       # Then test
```

### Error: "CUDA out of memory"

**Solution**: Your GPU doesn't have enough memory.

Try in `test_model.py`:
```python
# Use CPU instead
device = "cpu"

# Or reduce max_length
MAX_LENGTH = 50
```

### Output is Repetitive

**Solution**: Increase repetition penalty

```python
REPETITION_PENALTY = 1.5  # or higher
```

### Output is Too Random/Nonsensical

**Solution**: Lower temperature

```python
TEMPERATURE = 0.3  # More conservative
```

### Output is Too Boring/Generic

**Solution**: Increase temperature

```python
TEMPERATURE = 1.0  # More creative
```

---

## 💡 Tips for Best Results

1. **Test Early and Often**: Test with a few prompts during training checkpoints
2. **Use Relevant Prompts**: Test with prompts similar to your training data
3. **Compare Side-by-Side**: Always compare original vs fine-tuned to see the difference
4. **Adjust Parameters**: Experiment with temperature and top_p for different styles
5. **Document Results**: Keep track of what settings produce the best outputs

---

## 📚 Related Files

- `simple_finetune.py` - Main training script (includes automatic comparison)
- `test_model.py` - Interactive testing script
- `DEBUG_FP16_ISSUE.md` - Troubleshooting guide for training issues
- `README.md` - Complete project documentation

