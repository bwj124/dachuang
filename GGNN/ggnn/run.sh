#!/bin/bash

echo ------------------------- >> log_ggnn
echo train start:>>log_ggnn
date >> log_ggnn
./run_exp.sh
echo train end: >> log_ggnn
date >> log_ggnn
./run_test.sh
echo test end: >> log_ggnn
date >> log_ggnn
./generate_vectors.sh
echo generating vectors end: >> log_ggnn
date >> log_ggnn
echo ------------------------- >> log_ggnn
