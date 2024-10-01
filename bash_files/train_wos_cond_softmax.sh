#!/bin/bash

# Set the path to the config file
config_file="configs/aaaa_final_wos/vanilla_bert_wos_conditional_softmax.json"

# Run the training four times
for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=1 python3 train.py -cfg "$config_file" 

    # Set the path for predictions
    path="predictions_runs/wos/cond_softmax"

    # Rename files after each training iteration
    mv "${path}/prob.pickle" "${path}/prob_${i}.pickle"
    mv "${path}/logits.pickle" "${path}/logits_${i}.pickle"
    mv "${path}/labels.pickle" "${path}/labels_${i}.pickle"
done

# Set the path to the config file
config_file="configs/aaaa_final_wos/vanilla_bert_wos_leaf_softmax.json"

# Run the training four times
for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=1 python3 train.py -cfg "$config_file" 

    # Set the path for predictions
    path="predictions_runs/wos/leaf_softmax"

    # Rename files after each training iteration
    mv "${path}/prob.pickle" "${path}/prob_${i}.pickle"
    mv "${path}/logits.pickle" "${path}/logits_${i}.pickle"
    mv "${path}/labels.pickle" "${path}/labels_${i}.pickle"
done
