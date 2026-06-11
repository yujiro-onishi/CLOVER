#!/bin/bash
# dataset path
calvin_dataset_path='/workspace/CLOVER/calvin/dataset/calvin_debug_dataset'

subfix=`date "+%Y%m%d-%H%M"`
log_file="logs/training_"${subfix}".log"

export PYTHONPATH=/workspace/CLOVER:/workspace/CLOVER/visual_planner:$PYTHONPATH

torchrun --nnodes=1 --nproc_per_node=3 train/train_calvin.py \
    --vision_encoder vc1-base \
    --num_epochs 10 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 16 \
    --run_name feedback_policy_calvin_abc \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 0 \
    --learning_rate 1e-4 \
    --window_size 5 \
    2>&1 | tee ${log_file}
