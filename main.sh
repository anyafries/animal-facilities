#!/bin/bash

# Train the model
if [ $3 -eq 1 ]
then
    python train.py --farm $1
fi

# Extract features
python extract_features.py --farm $1 --split_set $2

# Evaluate the model
python eval.py --farm $1 --split_set $2

