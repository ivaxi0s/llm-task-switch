#!/bin/bash
# This script runs the pilot experiment
# test set: gigaword
# incontext dataset: rotten tomatoes

conda activate inctxt

MODEL_NAME="mistral-7b"
INCONTEXT="rotten_tomatoes"

# incontext: gigaword
echo "Incontext: $INCONTEXT"
for i in {0..16..2}; do
  echo "---Running with $i examples---"
  # python main.py --eval_data_name gigaword --incontext_data_name gigaword --num_examples $i --force_rerun
  python main.py \
    --eval_data_name gigaword \
    --incontext_data_name $INCONTEXT \
    --num_examples $i \
    --model_name $MODEL_NAME \
    --batchsize 50 \
    --gpu_id 0
done
