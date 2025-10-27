#!/bin/bash

echo "start...."

nhids=(16)
nouts=(16)
temporal_attention_layer_heads=(1)
heads=(1)
split_counts=(7)
dataset="enron184"

for split_count in ${split_counts[@]}
do
    for ddy_attention_layer_head in ${temporal_attention_layer_heads[@]}
    do
        for head in ${heads[@]}
        do
            for nout in ${nouts[@]}
            do
                for nhid in ${nhids[@]}
                do
                    python ./main_single.py --nhid=$nhid --nout=$nout --temporal_attention_layer_heads=$ddy_attention_layer_head --heads=$head --dataset=$dataset --device_id=3 --lr=0.01 --patience=40 --max_epoch=400 --split_count=$split_count
                done
            done
        done
    done
done
