#!/bin/bash
# This script runs the experiment
# test set: rotten tomatoes
# model: mistral-7b
# NOTE: The test size is limited to 100

conda activate inctxt

MODEL_NAME="mistral-7b"

INCONTEXT_SETS=("tweetqa" "gigaword" "mmluaa" "rotten_tomatoes")

# For each incontext dataset
for DATASET in "${INCONTEXT_SETS[@]}"; do
  echo "Incontext: $DATASET"
  for i in {0..6..3}; do
    echo "---Running with $i examples---"
    python likelihoods.py \
      --eval_data_name rotten_tomatoes \
      --incontext_data_name $DATASET \
      --num_examples $i \
      --model_name $MODEL_NAME \
      --batchsize 1 \
      --iterative \
      --likelihoods \
      --eval_size 100 \
      --gpu_id 2
    # --no_predict \
    # --force_rerun \
  done
done
