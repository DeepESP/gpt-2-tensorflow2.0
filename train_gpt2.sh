#!/usr/bin/env bash

num_layers=12
embedding_size=768
num_heads=12
dff=3072
max_seq_len=1024
vocab_size=50257
optimizer="adam"
batch_size=2
learning_rate=0.0001
dataset_path="/media/mega_disco/DataSets/GTP-2/TFRecord/V2/data_10240/*.tfrecord"
model_dir="/media/mega_disco/DataSets/GTP-2/Trained/tf2/model"
log_dir="/media/mega_disco/DataSets/GTP-2/Trained/tf2/log"

python train_gpt2.py \
    --num-layers=${num_layers} \
    --embedding-size=${embedding_size} \
    --num-heads=${num_heads} \
    --dff=${dff} \
    --max-seq-len=${max_seq_len} \
    --learning-rate=${learning_rate} \
    --vocab-size=${vocab_size} \
    --optimizer=${optimizer} \
    --batch-size=${batch_size} \
    --learning-rate=${learning_rate} \
    --distributed=False \
    --dataset-path=${dataset_path} \
    --model-dir=${model_dir} \
    --log-dir=${log_dir}
