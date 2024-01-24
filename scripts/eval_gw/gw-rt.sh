#!/bin/bash
# This script runs the experiment
# test set: gigaword
# incontext dataset: rotten tomatoes

conda activate inctxt

MODEL_NAME="mistral-7b"
INCONTEXT="rotten_tomatoes"

# incontext: gigaword
echo "Incontext: $INCONTEXT"
for i in {0..10..2}; do
  echo "---Running with $i examples---"
  # python main.py --eval_data_name gigaword --incontext_data_name gigaword --num_examples $i --force_rerun
  python main.py \
    --eval_data_name gigaword \
    --incontext_data_name $INCONTEXT \
    --num_examples $i \
    --model_name $MODEL_NAME \
    --batchsize 1 \
    --iterative \
    --gpu_id 1
done
