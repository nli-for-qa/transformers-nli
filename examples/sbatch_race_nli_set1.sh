#!/bin/bash
#SBATCH --job-name=race_nli-%j
#SBATCH -p 2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.out
#SBATCH --mem=50000
#module load nccl2
#srun python3 -m torch.distributed.launch \
#    --nproc_per_node 1 
python run_nli.py \
    --model_type roberta-nli \
    --model_name_or_path roberta-large \
    --task_name nli \
    --do_train \
    --do_lower_case \
    --data_dir .data/RACE/nli_set1 \
    --evaluate_during_training \
    --max_seq_length 512 \
    --logging_steps 500 \
    --save_steps 10000 \
    --per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --per_gpu_eval_batch_size 2 \
    --num_train_epochs 10.0 \
    --overwrite_output_dir \
    --output_dir temp \
    --warmup_steps 12000 \
    --wandb \
    --wandb_project nli-for-qa \
    --tags race_nli,single_gpu
