#!/bin/bash
# This script runs the pilot experiment
# test set: rotten tomatoes
# for the incontext dataset: gigaword

conda activate inctxt

# As a for loop 0:10 with 2 step size
for i in {0..10..2}; do
  echo "Running with $i examples"
  python main.py --incontext_data_name gigaword --num_examples $i --force_rerun
done
