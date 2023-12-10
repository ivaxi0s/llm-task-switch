#!/bin/bash
# This script runs the pilot experiment
# test set: rotten tomatoes
# for the incontext dataset: rotten tomatoes

# source ~/.bashrc
conda activate inctxt
# conda activate /home/ag2118/rds/hpc-work/envs/inctxt
# source scripts/activate_env.sh

# As a for loop
for i in {0..10}; do
  echo "Running with $i examples"
  python main.py --incontext_data_name rt --num_examples $i
done
