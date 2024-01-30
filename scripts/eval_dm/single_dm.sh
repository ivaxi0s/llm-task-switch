#!/bin/bash

# This script is for running experiments with
# MODEL_NAME
# TEST_SET: gigaword
# INCONTEXT: dailymail
# Use this like so: ./single_gw-dm.sh --num_examples 4

conda activate inctxt

# MODEL_NAME="mistral-7b"
INCONTEXT="dailymail"
MODEL_NAME="llama-7b"

# NUM_EXAMPLES=4 # Set this as an argument

# incontext: gigaword
echo "Incontext: $INCONTEXT"

echo -e "---Running with $@---"
python ../main.py \
  --eval_data_name dailymail \
  --incontext_data_name $INCONTEXT \
  --batchsize 1 \
  --eval_size 1000 \
  --model_name $MODEL_NAME \
  --force_rerun \
  --iterative \
  $@
# --model_name $MODEL_NAME \
# --num_examples $NUM_EXAMPLES \
# --gpu_id 2
