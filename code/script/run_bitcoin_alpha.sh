#!/bin/bash

echo "start...."

model=("dynhat")

seq_models=("Attention") # "Attention" "RNN" "LSTM" "GRU" BiLSTM
aggregations=("deg" "att")
manifolds=("Hyperboloid") #  "PoincareBall" "Euclidean"
split_counts=(13)
lrs=(0.01 0.001)
dataset="bitcoin_alpha"
fix_curvature=True

for split_count in ${split_counts[@]}
do
    for aggregation in ${aggregations[@]}
    do
        for manifold in ${manifolds[@]}
        do
            for seq_model in ${seq_models[@]}
            do
                for lr in ${lrs[@]}
                do
                    python ./main_single.py --aggregation=$aggregation --manifold=$manifold --fix_curvature=$fix_curvature --seq_model=$seq_model --lr=$lr --max_epoch=400 --split_count=$split_count --model=$model --dataset=$dataset
                done
            done
        done
    done
done
