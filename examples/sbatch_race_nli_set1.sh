#!/bin/bash
#SBATCH --job-name=race_nli-%j
#SBATCH -p m40-long
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.out
#SBATCH --mem=70000
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
    --logging_steps 1000 \
    --save_steps 10000 \
    --per_gpu_train_batch_size 6 \
    --gradient_accumulation_steps 5 \
    --learning_rate 5e-6 \
    --per_gpu_eval_batch_size 6 \
    --num_train_epochs 10.0 \
    --overwrite_output_dir \
    --output_dir temp \
    --warmup_steps 12000 \
    --max_grad_norm 1.0 \
    --wandb \
    --wandb_project nli-for-qa \
    --wandb_run_name race-nli-roberta-large-no-wt-decay \
    --seed 43 \
    --tags race_nli,roberta-large,single_gpu,no_wt_decay
