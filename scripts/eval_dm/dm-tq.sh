#!/bin/bash
# This script runs the experiment
# test set: gigaword
# incontext dataset: tweetqa

conda activate inctxt

MODEL_NAME="llama-7b"
INCONTEXT="tweetqa"

# incontext: gigaword
echo "Incontext: $INCONTEXT"
for i in {1..10..1}; do
  echo "---Running with $i examples---"
  # python main.py --eval_data_name gigaword --incontext_data_name gigaword --num_examples $i --force_rerun
  python main.py \
    --eval_data_name dailymail \
    --incontext_data_name $INCONTEXT \
    --num_examples $i \
    --model_name $MODEL_NAME \
    --batchsize 1 \
    --eval_size 1000 \
    --iterative \
    --force_rerun \
    --gpu_id 0
done
