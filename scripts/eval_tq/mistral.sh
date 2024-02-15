#!/bin/bash
# This script runs the experiment
# test set: tweetqa
# incontext dataset:
# model: llama-7b
# NOTE: The test size is *not* limited

conda activate inctxt

MODEL_NAME="mistral-7b"
INCONTEXT="mmluaa"

echo "Incontext: $INCONTEXT"
for i in {2..6..2}; do
  echo "---Running with $i examples---"
  python main.py \
    --eval_data_name tweetqa \
    --incontext_data_name $INCONTEXT \
    --num_examples $i \
    --model_name $MODEL_NAME \
    --batchsize 1 \
    --iterative \
    --gpu_id 2
  # --force_rerun \
done
