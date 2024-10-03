#!/bin/bash
# This script runs the experiment
# test set: mmluaa
# model: llama-7b
# NOTE: The test size is *not* limited

# NOTE: we use num_examples=6,
# we do not need to loop over the number of examples,
# because this is handled by the conversational.py script

conda activate inctxt

MODEL_NAME="llama-7b"

INCONTEXT_SETS=("mmluaa" "tweetqa" "gigaword" "rotten_tomatoes")

# For each incontext dataset
for INCONTEXT_SET in "${INCONTEXT_SETS[@]}"; do
  echo "Incontext: $INCONTEXT_SET"
  python ../conversational.py \
    --eval_data_name mmluaa \
    --incontext_data_name $INCONTEXT_SET \
    --num_examples 6 \
    --model_name $MODEL_NAME \
    --batchsize 1 \
    --iterative \
    --gpu_id 0
  # --no_predict \
  # --force_rerun \
done
