#!/bin/bash
# This script runs the experiment
# test set: mmluaa
# model: mistral-7b
# NOTE: The test size is *not* limited

# NOTE: we use num_examples=6,
# we do not need to loop over the number of examples,
# because this is handled by the random_conversation.py script

conda activate inctxt

MODEL_NAME="mistral-7b"

# For each incontext dataset
python ../random_conversation.py \
  --eval_data_name mmluaa \
  --incontext_data_name mmluaa \
  --num_examples 6 \
  --model_name $MODEL_NAME \
  --batchsize 1 \
  --iterative \
  --gpu_id 0
# --no_predict \
# --force_rerun \
