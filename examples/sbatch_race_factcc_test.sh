#!/bin/bash
#SBATCH --job-name=ra_qa
#SBATCH -p titanx-short
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.out
#SBATCH --mem=80GB
module load python3/3.7.3-1904
module load nccl2
module load cuda101
cd /mnt/nfs/scratch1/dhruveshpate/nli_for_qa/pytorch-transformers/examples
source /mnt/nfs/scratch1/dhruveshpate/nli_for_qa/pytorch-transformers/examples/.venv_examples/bin/activate

export OMP_NUM_THREADS=2
python run_qa_as_nli_eval.py \
    --model_type roberta-nli-transferable \
    --model_name_or_path "/mnt/nfs/scratch1/dhruveshpate/nli_for_qa/models/RACE/qa/"\
    --task_name single_choice \
    --num_choices 1 \
    --do_lower_case \
    --data_dir /mnt/nfs/scratch1/dhruveshpate/nli_for_qa/qa-to-nli/.data/factCC \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size 4 \
    --save_preds \
    --threshold 0.99871 \
    --use_threshold \
    --test \
    --output_dir temp \
    --wandb \
    --wandb_entity "ibm-cs696ds-2020" \
    --wandb_project nli4qa \
    --wandb_run_name race-full-qa-factcc-test \
    --tags race-full,qa,roberta-large,mc-transferable,test

