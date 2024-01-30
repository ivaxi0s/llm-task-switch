#!/bin/bash
# This script runs the experiment
# test set: gigaword
# incontext dataset:
# model: llama-7b
# NOTE: The test size is *not* limited

conda activate inctxt

MODEL_NAME="mistral-7b"

# do not include dailmail, as we have to limit eval_Size
EVAL_SETS=("tweetqa" "gigaword")

# For each incontext dataset
for EVAL_SET in "${EVAL_SETS[@]}"; do
  echo "Incontext: $EVAL_SET"
  for i in {0..10..2}; do
    echo "---Running with $i examples---"
    python main.py \
      --eval_data_name $EVAL_SET \
      --incontext_data_name rotten_tomatoes \
      --num_examples $i \
      --model_name $MODEL_NAME \
      --batchsize 1 \
      --iterative \
      --force_rerun \
      --gpu_id 1
  done
done

# echo "Incontext: $INCONTEXT"
# for i in {0..10..2}; do
#   echo "---Running with $i examples---"
#   python main.py \
#     --eval_data_name rotten_tomatoes \
#     --incontext_data_name $INCONTEXT \
#     --num_examples $i \
#     --model_name $MODEL_NAME \
#     --batchsize 1 \
#     --iterative \
#     --force_rerun \
#     --gpu_id 3
# done
