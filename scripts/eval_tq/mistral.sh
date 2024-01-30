#!/bin/bash
# This script runs the experiment
# test set: tweetqa
# incontext dataset:
# model: mistral-7b

conda activate inctxt

MODEL_NAME="mistral-7b"
INCONTEXT="tweetqa"

echo "Incontext: $INCONTEXT"
for i in {0..10..1}; do
  echo "---Running with $i examples---"
  python main.py \
    --eval_data_name tweetqa \
    --incontext_data_name $INCONTEXT \
    --num_examples $i \
    --model_name $MODEL_NAME \
    --batchsize 1 \
    --iterative \
    --force_rerun \
    --gpu_id 1
done
