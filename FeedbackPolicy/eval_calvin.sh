#!/bin/bash
export EVALUTION_ROOT=$(pwd)

# Set CALVIN path
calvin_dataset_path='/workspace/CLOVER/calvin/dataset/calvin_debug_dataset'
calvin_conf_path="/workspace/CLOVER/calvin/calvin_models/conf"

# Set checkpoints path
visual_planner_checkpoint='/workspace/CLOVER/models/model-10.pt'
policy_checkpoint='/workspace/CLOVER/FeedbackPolicy/checkpoint_final_weights.pth'

export MESA_GL_VERSION_OVERRIDE=4.1
node_num=4


torchrun --nnodes=1 --nproc_per_node=${node_num} --master_port=6600 eval/eval_calvin.py \
    --visual_planner_checkpoint ${visual_planner_checkpoint} \
    --policy_checkpoint ${policy_checkpoint} \
    --calvin_dataset ${calvin_dataset_path} \
    --calvin_conf_path ${calvin_conf_path} \
    --sample_step 20 \


