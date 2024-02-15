#!/bin/bash
# This script runs the experiment
# test set: mmluaa
# incontext dataset:
# model: mistral-7b
# NOTE: The test size is *not* limited

conda activate inctxt

MODEL_NAME="mistral-7b"

# do not include dailmail, as we have to limit eval_Size
INCONTEXT_SETS=("dailymail" "rotten_tomatoes" "tweetqa" "gigaword")

# For each incontext dataset
for INCONTEXT_SET in "${INCONTEXT_SETS[@]}"; do
  echo "Incontext: $INCONTEXT_SET"
  for i in {0..6..1}; do
    echo "---Running with $i examples---"
    python main.py \
      --eval_data_name dailymail \
      --incontext_data_name $INCONTEXT_SET \
      --num_examples $i \
      --model_name $MODEL_NAME \
      --batchsize 1 \
      --iterative \
      --force_rerun \
      --no_predict \
      --eval_size 1000 \
      --gpu_id 0
    # --likelihoods \
  done
done
