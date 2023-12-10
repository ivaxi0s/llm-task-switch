#!/bin/bash
# This script runs the pilot experiment
# test set: rotten tomatoes
# different in-context example datasets
# num examples = 0
# Hence, the results should be exactly the same!

conda activate inctxt

echo "Running with 0 examples"

# Create a list of in-context datasets
incontext_data_names=(rt gigaword)

# As a for loop
for i in ${!incontext_data_names[@]}; do
  echo "Running with ${incontext_data_names[$i]} examples"
  python main.py --incontext_data_name ${incontext_data_names[$i]} --num_examples 0 --force_rerun
done
