#!/bin/bash
python run_nli.py \
    --model_type roberta-nli \
    --model_name_or_path roberta-large \
    --task_name nli \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir .data/RACE/test_set1 \
    --evaluate_during_training \
    --max_seq_length 256 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --overwrite_output_dir \
    --output_dir temp \
    --wandb \
    --wandb_project nli-for-qa \
    --tags test,local
