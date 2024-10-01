#!/bin/bash

# Set the path to the config file
config_file="configs/aaaa_final_wos/tin_bert_wos.json"

# Run the training four times
for i in {3..3}
do
    CUDA_VISIBLE_DEVICES=2 python3 train.py -cfg "$config_file" 

    # Set the path for predictions
    path="predictions_runs/wos/hitin"

    # Rename files after each training iteration
    mv "${path}/prob.pickle" "${path}/prob_${i}.pickle"
    mv "${path}/logits.pickle" "${path}/logits_${i}.pickle"
    mv "${path}/labels.pickle" "${path}/labels_${i}.pickle"
done
