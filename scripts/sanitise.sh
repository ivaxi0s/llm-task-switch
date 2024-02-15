#!/bin/bash
# This script runs the experiment
# test set: mmluaa
# incontext dataset:
# model: mistral-7b
# NOTE: The test size is *not* limited

conda activate inctxt

MODEL_NAME="gpt3.5"

# do not include dailmail, as we have to limit eval_Size
INCONTEXT_SETS=("mmluaa" "rotten_tomatoes")

# For each incontext dataset
for INCONTEXT_SET in "${INCONTEXT_SETS[@]}"; do
  echo "Incontext: $INCONTEXT_SET"
  for i in {0..6..1}; do
    echo "---Running with $i examples---"
    python sanitise_prompts.py \
      --eval_data_name gigaword \
      --incontext_data_name $INCONTEXT_SET \
      --num_examples $i \
      --model_name $MODEL_NAME \
      --batchsize 1 \
      --iterative \
      --eval_size 1000 \
      --gpu_id 0
    # --likelihoods \
    # --force_rerun \
    # --no_predict \
  done
done
