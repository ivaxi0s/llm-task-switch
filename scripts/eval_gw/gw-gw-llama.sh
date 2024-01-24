#!/bin/bash
# This script runs the pilot experiment
# test set: gigaword
# incontext dataset: gigaword, then rotten tomatoes

conda activate inctxt

MODEL_NAME="mistral-7b"

# incontext: gigaword
echo "Incontext: gigaword"
for i in {0..10..2}; do
  echo "---Running with $i examples---"
  # python main.py --eval_data_name gigaword --incontext_data_name gigaword --num_examples $i --force_rerun
  python main.py \
    --eval_data_name gigaword \
    --incontext_data_name gigaword \
    --num_examples $i \
    --model_name $MODEL_NAME \
    --force_rerun \
    --iterative \
    --batchsize 1 \
    --gpu_id 1
done

# incontext: rotten tomatoes
# As a for loop 0:20 with 10 step size
# echo "Incontext: rotten tomatoes"
# for i in {0..20..10}; do
#   echo "---Running with $i examples---"
#   python main.py --eval_data_name gigaword --incontext_data_name rt --num_examples $i
#   # python main.py --eval_data_name gigaword --incontext_data_name rt --num_examples $i --force_rerun
# done
