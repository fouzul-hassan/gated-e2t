#!/usr/bin/env bash
set -e

echo "🔹 Updating pip"
python3 -m pip install --upgrade pip

# echo "🔹 Cloning GLIM repo"
# git clone https://github.com/fouzul-hassan/energy-gated-glim.git
# cd gated-e2t

echo "🔹 Installing requirements"
pip install -r requirements.txt

echo "🔹 Hugging Face login"
huggingface-cli login --token "${HF_TOKEN}"

echo "🔹 Weights & Biases login"
wandb login "${WANDB_API_KEY}"

echo "Environment setup completed"
