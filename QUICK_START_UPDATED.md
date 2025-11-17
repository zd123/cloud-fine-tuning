# 🚀 Quick Start - Updated Version

## What's New?

Your cloud fine-tuning setup now has **powerful testing capabilities**!

### ✨ Two New Features

1. **Automatic Comparison** - See original vs fine-tuned model side-by-side
2. **Interactive Testing** - Test your model with any prompt you want

Plus, we fixed the FP16 gradient scaling bug! 🎉

---

## 📤 Step 1: Upload Updated Files

From your **LOCAL machine** (not Lambda):

```bash
cd ~/fine-turning-lecture-2025/cloud-fine-tuning

# Upload the new/updated files
scp simple_finetune.py test_model.py ubuntu@<YOUR-LAMBDA-IP>:~/cloud-fine-tuning/
```

Replace `<YOUR-LAMBDA-IP>` with your actual Lambda instance IP.

---

## 🏃 Step 2: Run Training

SSH into Lambda and run the updated script:

```bash
ssh ubuntu@<YOUR-LAMBDA-IP>
cd ~/cloud-fine-tuning
source venv/bin/activate

# Run training (this now includes automatic comparison!)
python simple_finetune.py
```

### What You'll See (New!)

After training completes, the script will automatically:

```
============================================================
STEP 7: Comparing models (before vs after fine-tuning)...
============================================================

Loading original model for comparison...
✓ Original model loaded

============================================================
TEST 1: 'Once upon a time'
============================================================

🔵 ORIGINAL MODEL OUTPUT:
------------------------------------------------------------
Once upon a time there was a kingdom in a faraway land...

🟢 FINE-TUNED MODEL OUTPUT:
------------------------------------------------------------
Once upon a time in the Bronx, there was a girl named Cardi
who had dreams bigger than the streets she grew up on...
------------------------------------------------------------
```

**Notice the difference?** Your fine-tuned model adopted the style of your training data!

---

## 🎮 Step 3: Interactive Testing

Now try the interactive testing script:

```bash
# Still in your Lambda SSH session:
python test_model.py
```

### Example Session

```
🤖 FINE-TUNED MODEL TESTER
======================================================================
✓ GPU detected: NVIDIA A10
✓ Model loaded successfully (494M parameters)

🎯 INTERACTIVE MODE
======================================================================

💬 Enter prompt (or 'quit' to exit): Tell me about New York

🔄 Generating...

🤖 Generated Text:
------------------------------------------------------------
Tell me about New York, the city that never sleeps, where
dreams are made and the hustle is real. From the Bronx to
Brooklyn, Manhattan to Queens, every borough got its own
flavor, its own story...
------------------------------------------------------------

💬 Enter prompt (or 'quit' to exit): What makes music great?

🔄 Generating...

🤖 Generated Text:
------------------------------------------------------------
What makes music great is the beat, the flow, the energy
you bring to the mic. It's about telling your truth and
connecting with people who feel what you feel...
------------------------------------------------------------

💬 Enter prompt (or 'quit' to exit): quit
👋 Goodbye!
```

---

## 🎯 Quick Command Reference

```bash
# Upload files from local machine
scp simple_finetune.py test_model.py ubuntu@<IP>:~/cloud-fine-tuning/

# SSH into Lambda
ssh ubuntu@<IP>

# Activate environment
cd ~/cloud-fine-tuning
source venv/bin/activate

# Run training (includes automatic comparison)
python simple_finetune.py

# Test interactively
python test_model.py

# Test with one prompt (command-line mode)
python test_model.py "Your prompt here"
```

---

## ⚙️ Customization

### Change Test Prompts (in `simple_finetune.py`)

Edit around line 204:

```python
test_prompts = [
    "Once upon a time",
    "Tell me about",
    "In my opinion,",
    "Your custom prompt here",  # Add more!
]
```

### Adjust Generation Settings (in `test_model.py`)

Edit at the top of the file:

```python
MAX_LENGTH = 150              # Make longer/shorter
TEMPERATURE = 0.7             # 0.0 = boring, 1.0 = wild
TOP_P = 0.9                   # Nucleus sampling
REPETITION_PENALTY = 1.1      # Avoid repetition
```

**Try these presets:**

**Creative Writing:**
```python
MAX_LENGTH = 300
TEMPERATURE = 0.9
TOP_P = 0.95
```

**Conservative/Factual:**
```python
MAX_LENGTH = 100
TEMPERATURE = 0.3
TOP_P = 0.8
```

---

## 📚 Documentation Files

We created comprehensive guides for you:

- **`README.md`** - Main documentation (updated with testing section)
- **`TESTING_GUIDE.md`** - Complete guide to testing your model
- **`DEBUG_FP16_ISSUE.md`** - Explains the bug fix in detail
- **`CHANGELOG.md`** - All changes and improvements
- **`QUICK_START_UPDATED.md`** - This file!

---

## 🎓 Teaching Tips

### Show Students the Difference

The automatic comparison is perfect for demonstrating:
- ✅ **Style Transfer**: Model adopts training data style
- ✅ **Vocabulary**: Uses words from training corpus
- ✅ **Topic Adaptation**: Stays on-theme

### Live Demo

Run `test_model.py` in class and let students:
- Suggest prompts to test
- See real-time generation
- Compare different temperature settings
- Understand generation parameters

### Assignments

1. **Compare Models**: Have students analyze original vs fine-tuned outputs
2. **Parameter Tuning**: Adjust temperature/top_p and observe differences
3. **Custom Prompts**: Design prompts that showcase fine-tuning impact

---

## 🐛 Troubleshooting

### "Model not found"
Run training first: `python simple_finetune.py`

### "CUDA out of memory"
Lower `MAX_LENGTH` in `test_model.py` or use CPU

### Output is repetitive
Increase `REPETITION_PENALTY = 1.5`

### Output is too random
Lower `TEMPERATURE = 0.3`

### Still have FP16 error?
Make sure you uploaded the updated `simple_finetune.py`
```bash
ssh ubuntu@<IP>
cd ~/cloud-fine-tuning
grep "bf16" simple_finetune.py
# Should see bf16 configuration, not just fp16
```

---

## 💡 Pro Tips

1. **Test Early**: Run a few test prompts while training to verify style adoption
2. **Document Results**: Keep track of which settings work best
3. **Try Edge Cases**: Test prompts that weren't in training data
4. **Batch Test**: Create a file of prompts and test them all
5. **Compare Checkpoints**: Save checkpoints and compare early vs late training

---

## 🎉 Summary

You now have:
- ✅ Fixed FP16 gradient scaling bug (no more crashes!)
- ✅ Automatic before/after comparison (built into training)
- ✅ Interactive testing script (`test_model.py`)
- ✅ Comprehensive documentation (5 new guides!)
- ✅ Well-commented code (great for teaching)

**Next step:** Upload the files and run training! 🚀

```bash
# From local machine:
cd ~/fine-turning-lecture-2025/cloud-fine-tuning
scp simple_finetune.py test_model.py ubuntu@<YOUR-IP>:~/cloud-fine-tuning/

# Then SSH and run:
ssh ubuntu@<YOUR-IP>
cd ~/cloud-fine-tuning
source venv/bin/activate
python simple_finetune.py
```

Happy fine-tuning! 🎊

