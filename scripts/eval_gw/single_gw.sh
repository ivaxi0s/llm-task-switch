#!/bin/bash

# This script is for running experiments with
# MODEL_NAME
# TEST_SET: gigaword
# INCONTEXT: dailymail
# Use this like so: ./single_gw-dm.sh --num_examples 4

conda activate inctxt

INCONTEXT="dailymail"
MODEL_NAME="llama-7b"
# NUM_EXAMPLES=4 # Set this as an argument

# incontext: gigaword
echo "Incontext: $INCONTEXT"

echo "---Running with $NUM_EXAMPLES examples---"
python ../main.py \
  --eval_data_name gigaword \
  --model_name $MODEL_NAME \
  --incontext_data_name $INCONTEXT \
  --batchsize 1 \
  --iterative \
  $@
# --model_name $MODEL_NAME \
# --num_examples $NUM_EXAMPLES \
# --gpu_id 2
