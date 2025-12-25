#!/bin/bash

# DiffFinance Quick Start Script

echo "=================================="
echo "DiffFinance Quick Start"
echo "=================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python --version

# Install dependencies
echo -e "\n2. Installing dependencies..."
pip install -r requirements.txt

# Run quick start examples
echo -e "\n3. Running quick start examples..."
python examples/quickstart.py

# Train a small model (optional)
echo -e "\n4. Training a small model (this may take a few minutes)..."
python training/train_market_diffusion.py \
    --num_episodes 50 \
    --episode_length 50 \
    --batch_size 32 \
    --num_epochs 5 \
    --device cpu

# Run simulation
echo -e "\n5. Running market simulation..."
python inference/simulate_market.py \
    --num_agents 4 \
    --num_steps 100 \
    --device cpu

# Evaluate strategies
echo -e "\n6. Evaluating strategies..."
python evaluation/evaluate_strategies.py \
    --num_agents 4 \
    --num_episodes 10 \
    --device cpu

echo -e "\n=================================="
echo "Quick start completed!"
echo "=================================="
echo -e "\nCheck the following directories for results:"
echo "  - checkpoints/: Trained models"
echo "  - results/: Simulation results"
echo "  - evaluation/: Evaluation metrics"

