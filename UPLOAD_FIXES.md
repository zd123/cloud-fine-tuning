# Upload Fixed Script to Lambda Instance

## The Problem

The Lambda instance is still running the OLD version of `simple_finetune.py` that has the FP16 bug.
The fixes were applied to your LOCAL copy, but haven't been transferred to the cloud yet.

## Solution: Upload the Fixed Script

### Option 1: Use SCP to Upload the Fixed File (Fastest)

From your **local terminal** (not SSH'd into Lambda):

```bash
# Navigate to the cloud-fine-tuning directory
cd ~/fine-turning-lecture-2025/cloud-fine-tuning

# Upload the fixed script to Lambda
scp simple_finetune.py ubuntu@<YOUR-LAMBDA-IP>:~/cloud-fine-tuning/

# Example:
# scp simple_finetune.py ubuntu@150.136.14.72:~/cloud-fine-tuning/
```

### Option 2: Pull Latest Changes from Git

If you've committed the changes to git:

```bash
# SSH into your Lambda instance
ssh ubuntu@<YOUR-LAMBDA-IP>

# Navigate to the repo
cd ~/cloud-fine-tuning

# Pull the latest changes
git pull origin main

# Verify the update worked
grep -n "bf16" simple_finetune.py
# You should see bf16 mentioned in the output
```

### Option 3: Re-clone the Entire Repository

If you want a fresh copy:

```bash
# SSH into your Lambda instance
ssh ubuntu@<YOUR-LAMBDA-IP>

# Remove the old directory
rm -rf ~/cloud-fine-tuning

# Clone again
git clone <your-repo-url> ~/cloud-fine-tuning
cd ~/cloud-fine-tuning

# Set up and run
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python simple_finetune.py
```

## Verify the Fix is Applied

After uploading, check that the new version is being used:

```bash
# SSH into Lambda
ssh ubuntu@<YOUR-LAMBDA-IP>

# Check for the bf16 configuration (new code)
cd ~/cloud-fine-tuning
grep -A 2 "bf16" simple_finetune.py
```

You should see:
```python
bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
fp16=False,  # Explicitly disable fp16 to avoid gradient scaling conflicts
```

## Now Run the Fixed Script

```bash
# Make sure you're in the right directory
cd ~/cloud-fine-tuning
source venv/bin/activate

# Run the fixed version
python simple_finetune.py
```

## What You'll See (Success Signs)

✅ **No more**: "Attempting to unscale FP16 gradients" error
✅ **No more**: "tokenizer is deprecated" warning (now uses `processing_class`)
✅ Training progress bar starts and runs
✅ You'll see: "bf16": true in the training arguments

## Quick Reference

```bash
# From your LOCAL machine:
cd ~/fine-turning-lecture-2025/cloud-fine-tuning
scp simple_finetune.py ubuntu@<LAMBDA-IP>:~/cloud-fine-tuning/

# Then SSH into Lambda and run:
ssh ubuntu@<LAMBDA-IP>
cd ~/cloud-fine-tuning
source venv/bin/activate
python simple_finetune.py
```

