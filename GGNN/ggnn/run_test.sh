#!/bin/bash


# Usage
# ./run_test.sh APIReplacement Mkdir

# --restore logs/rw-slice-rand-train-z/rw-slice-rand-train-z.json_model_best.pickle

python3 ggnn_sparse_ggnn.py --predict --restore logs/rw-slice-rand-train-z/rw-slice-rand-train-z.json_model_best.pickle --data_dir data/mine --valid_file rw-slice-rand-test-z.json --train_file rw-slice-rand-train-z.json
