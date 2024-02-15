#!/bin/bash
# This script runs the experiment
# test set: gigaword
# incontext dataset:
# model: llama-7b
# NOTE: The test size is *not* limited

conda activate inctxt

MODEL_NAME="mistral-7b"

# do not include dailmail, as we have to limit eval_Size
INCONTEXT_SETS=("mmluaa")

# For each incontext dataset
for INCTXT_SET in "${INCONTEXT_SETS[@]}"; do
  echo "Incontext: $INCTXT_SET"
  for i in {0..6..1}; do
    echo "---Running with $i examples---"
    python main.py \
      --eval_data_name rotten_tomatoes \
      --incontext_data_name $INCTXT_SET \
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
