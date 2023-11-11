#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/data1/haihongzhao/DSAA6000I-Final-Project-Group-7/training_output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

nohup deepspeed --include localhost:4,5,6,7 --master_port=27000 main_group_6.py \
   --data_path local/jsonfile \
   --data_output_path /data1/haihongzhao/DSAA6000I-Final-Project-Group-7/data_output_path/bz4 \
   --data_split 10,0,0 \
   --model_name_or_path /data2/Llama-2-7b-hf \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 1024 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --num_train_epochs 5  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 8 \
   --only_optimize_lora \
   --lora_module_name "self_attn." \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log &