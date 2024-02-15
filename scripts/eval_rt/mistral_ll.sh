#!/bin/bash
# This script runs the experiment
# test set: gigaword
# incontext dataset:
# model: llama-7b
# NOTE: The test size is *not* limited

conda activate inctxt

MODEL_NAME="mistral-7b"

# do not include dailmail, as we have to limit eval_Size
INCONTEXT_SETS=("tweetqa" "gigaword" "mmluaa" "rotten_tomatoes")

# For each incontext dataset
for DATASET in "${INCONTEXT_SETS[@]}"; do
  echo "Incontext: $DATASET"
  for i in {0..6..2}; do
    echo "---Running with $i examples---"
    python likelihoods.py \
      --eval_data_name rotten_tomatoes \
      --incontext_data_name $DATASET \
      --num_examples $i \
      --model_name $MODEL_NAME \
      --batchsize 1 \
      --iterative \
      --no_predict \
      --likelihoods \
      --eval_size 100 \
      --gpu_id 0
    # --force_rerun \
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
