#!/bin/bash
# This script runs the experiment
# test set: mmluaa
# model: llama-7b
# NOTE: The test size is *not* limited

conda activate inctxt

MODEL_NAME="llama-7b"

INCONTEXT_SETS=("mmluaa" "rotten_tomatoes" "tweetqa" "gigaword")
SEEDS=(100 1000 10000)
# For each seed
for SEED in "${SEEDS[@]}"; do
  echo "Seed: $SEED"
  # For each incontext dataset
  for INCONTEXT_SET in "${INCONTEXT_SETS[@]}"; do
    echo "Incontext: $INCONTEXT_SET"
    for i in {0..6..3}; do
      echo "---Running with $i examples---"
      python ../main.py \
        --eval_data_name mmluaa \
        --incontext_data_name $INCONTEXT_SET \
        --num_examples $i \
        --model_name $MODEL_NAME \
        --batchsize 1 \
        --iterative \
        --gpu_id 1 \
        --seed $SEED
      # --force_rerun \
      # --no_predict \
    done
  done
done
