#!/bin/bash

# ./run_exp.sh APIReplacement Mkdir 0

python3 ggnn_sparse_ggnn.py --data_dir data/mine --valid_file rw-slice-rand-valid-z.json --train_file rw-slice-rand-train-z.json
