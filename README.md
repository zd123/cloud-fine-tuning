# Cloud Fine-Tuning Lab

A simple blueprint for fine-tuning language models on Lambda Labs GPU instances with local development workflow.

## 🚀 Quick Start: Lambda Labs GPU Setup

### Step 1: Launch Lambda Labs Instance

1. **Sign up/Login** at [lambdalabs.com](https://lambdalabs.com/service/gpu-cloud)
2. **Click "Launch Instance"**
3. **Select GPU**: Choose **1x A10 (24 GB)** (~$0.60/hr) - good balance of speed/cost
4. **Select Region**: Pick one with availability (usually closest to you)
5. **Select Instance Type**: Keep default
6. **SSH Key**: 
   - If you already have an SSH key: Select it
   - If not: Click "Add SSH Key" → Give it a name → Paste your public key
   - **Don't have an SSH key?** See "Generate SSH Key" section below
7. **Filesystem**: Select **"Don't attach a file system"** (ephemeral is fine for development)
8. **Click "Launch Instance"**
9. **Wait ~60 seconds** for instance to start
10. **Copy the IP address** shown (e.g., `150.136.14.72`)

---

### Step 2: Connect to Your Lambda Instance

From your **local terminal** (Mac/Linux):

```bash
# Connect via SSH (Lambda uses 'ubuntu' as default user)
ssh ubuntu@<YOUR-LAMBDA-IP>

# Example:
# ssh ubuntu@150.136.14.72
```

**First time connecting?** You'll see a message asking to verify the fingerprint:
- Type `yes` and press Enter

**Connection refused?** Wait another 30 seconds - instance may still be starting.

---

### 📋 Generate SSH Key (if you don't have one)

If you don't have an SSH key yet:

```bash
# On your LOCAL machine (Mac/Linux terminal):
ssh-keygen -t ed25519 -C "your_email@example.com"

# Press Enter for default location (~/.ssh/id_ed25519)
# Press Enter twice for no passphrase (or set one if you prefer)

# Copy your PUBLIC key:
cat ~/.ssh/id_ed25519.pub

# Copy the entire output and paste it into Lambda Labs when adding SSH key
```

**On Windows?** Use Git Bash, WSL, or PowerShell:
```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
type ~\.ssh\id_ed25519.pub
```

---

### Step 3: Set Up Environment on Lambda

**Once connected via SSH**, run these commands:

```bash
# Clone your repo (replace with your repo URL)
git clone https://github.com/YOUR-USERNAME/cloud-fine-tuning.git
cd cloud-fine-tuning

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch transformers datasets accelerate jupyter

# Verify GPU is available
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"

# Should output:
# GPU Available: True
# GPU Name: NVIDIA A10
```

---

### Step 4: Add Training Data

You have **two options** to get your training data onto Lambda:

#### Option A: Upload from Local Machine

From your **local terminal** (new window, not SSH session):

```bash
# Upload your training data files
scp training_data/*.txt ubuntu@<LAMBDA-IP>:~/cloud-fine-tuning/training_data/

# Example:
# scp training_data/*.txt ubuntu@150.136.14.72:~/cloud-fine-tuning/training_data/
```

#### Option B: Create/Edit Files Directly on Lambda

```bash
# While connected via SSH:
cd training_data/
nano sample.txt  # or vim/vi
# Type or paste your training text
# Save and exit (Ctrl+X, then Y, then Enter for nano)
```

---

### Step 5: Run Fine-Tuning

#### Option A: Simple Blueprint Script (Recommended for Testing)

```bash
# While SSH'd into Lambda, in your repo directory:
source venv/bin/activate  # If not already activated
python simple_finetune.py

# This will:
# - Load your data
# - Train the model
# - Show you progress in real-time
# - Save the trained model
```

#### Option B: Full Jupyter Notebook

```bash
# Start Jupyter (accessible from your local machine)
jupyter lab --no-browser --ip=0.0.0.0 --port=8888

# You'll see output like:
#   http://0.0.0.0:8888/lab?token=abc123def456...
# 
# Copy the URL and replace 0.0.0.0 with your Lambda IP:
#   http://150.136.14.72:8888/lab?token=abc123def456...
#
# Paste that into your local browser OR connect via VS Code (see Step 6)
```

---

### Step 6: Connect Local VS Code to Lambda Jupyter (Optional)

This lets you **edit locally** but **execute on Lambda's GPU**:

1. **Keep Jupyter running on Lambda** (from Step 5B)
2. **In VS Code on your local machine**:
   - Install "Jupyter" extension
   - Open your local copy of `fine_tuning_lab.ipynb`
   - Click "Select Kernel" (top right)
   - Choose "Existing Jupyter Server"
   - Enter: `http://<LAMBDA-IP>:8888/?token=<YOUR-TOKEN>`
3. **Run cells** - they execute on Lambda's GPU!

---

### Step 7: Download Your Trained Model

After training completes:

```bash
# From your LOCAL machine (new terminal):
scp -r ubuntu@<LAMBDA-IP>:~/cloud-fine-tuning/output/final_model ./my_model/

# This downloads the trained model to your local machine
```

---

### Step 8: Terminate Instance (Save Money!)

**Don't forget to terminate when done!**

1. Go to Lambda Labs dashboard
2. Click "Terminate" on your instance
3. Confirm termination

**Cost**: You're charged per second, so terminate ASAP when not using!

---

## ⚡ Quick Command Reference

```bash
# Connect to Lambda
ssh ubuntu@<LAMBDA-IP>

# Activate virtual environment (do this every time you connect)
cd cloud-fine-tuning
source venv/bin/activate

# Run training
python simple_finetune.py

# Start Jupyter
jupyter lab --no-browser --ip=0.0.0.0 --port=8888

# Upload files from local to Lambda
scp file.txt ubuntu@<LAMBDA-IP>:~/cloud-fine-tuning/

# Download files from Lambda to local
scp ubuntu@<LAMBDA-IP>:~/cloud-fine-tuning/output/model.bin ./

# Check GPU usage (while training)
nvidia-smi

# Exit SSH
exit
```

---

## 🐛 Troubleshooting Connection Issues

### "Connection refused"
- **Solution**: Wait 1-2 minutes after launching - instance is still starting
- Check instance status in Lambda dashboard (should be "running")

### "Permission denied (publickey)"
- **Solution**: Your SSH key isn't configured correctly
- Re-add your SSH key in Lambda dashboard: Settings → SSH Keys
- Make sure you copied the `.pub` (public) key, not the private key

### "Could not resolve hostname"
- **Solution**: Check you're using the correct IP address from Lambda dashboard
- IP addresses change each time you launch a new instance

### Can't access Jupyter in browser
- **Solution**: Make sure you're using Lambda's IP, not `0.0.0.0` or `localhost`
- Check port 8888 in the URL
- Include the token from Jupyter's output

### GPU not detected
- **Solution**: Make sure you selected a GPU instance (not CPU)
- Try: `nvidia-smi` to verify GPU is present

### Training crashes with "Attempting to unscale FP16 gradients"
- **Solution**: This was a known issue with FP16/BF16 configuration that has been fixed
- The script now uses BFloat16 (`bf16`) instead of FP16 for better stability
- See `DEBUG_FP16_ISSUE.md` for detailed explanation of the problem and solution
- If you still encounter this, your GPU might not support BF16 - try disabling mixed precision

---

## 📁 Repository Structure

```
cloud-fine-tuning/
├── README.md                    # This file
├── simple_finetune.py          # Minimal fine-tuning script
├── DEBUG_FP16_ISSUE.md         # Troubleshooting guide for gradient scaling error
├── .gitignore                  # Ignore venv, outputs, etc.
└── training_data/              # Your .txt files go here
    └── sample.txt
```

---

## 💡 Workflow Summary

1. **Launch** Lambda Labs GPU instance
2. **SSH** into the instance
3. **Clone** this repo and set up virtual environment
4. **Upload** training data via `scp`
5. **Run** `simple_finetune.py` or Jupyter notebook
6. **Download** trained model
7. **Terminate** instance to stop charges

---

## 📝 Next Steps

After you've successfully run the simple blueprint:

- Try the full notebook: `fine_tuning_lab.ipynb` (see parent repo)
- Experiment with different hyperparameters
- Train on your own custom datasets
- Try larger models (if you have more GPU memory)

---

## 💰 Cost Estimates

| GPU Type | Memory | Cost/Hour | Typical Training Time |
|----------|--------|-----------|----------------------|
| A10      | 24 GB  | $0.60     | 10-20 minutes        |
| A6000    | 48 GB  | $0.80     | 10-20 minutes        |
| A100     | 40 GB  | $1.29     | 5-10 minutes         |

**Example cost**: Training on A10 for 15 minutes = $0.15

---

## 🔒 Security Notes

- **Never commit** your SSH private key to Git
- **Don't share** your Lambda IP publicly (temporary anyway)
- **Terminate instances** when not in use
- **Use `.gitignore`** to avoid committing large model files

---

## 🤝 Support

Having issues? Check:
1. Troubleshooting section above
2. Lambda Labs documentation: [docs.lambdalabs.com](https://docs.lambdalabs.com)
3. HuggingFace forums: [discuss.huggingface.co](https://discuss.huggingface.co)

---

**Happy Fine-Tuning!** 🚀
