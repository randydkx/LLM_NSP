#!/bin/bash
# Continual Learning NS-GRPO Training Script
# This script demonstrates how to run continual learning with NS-GRPO

set -e


# 获取脚本所在的绝对目录，假设数据文件相对于项目根目录存放
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/..}" 


# --- WandB 配置 ---
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb_logs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_ROOT}/wandb_cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${PROJECT_ROOT}/wandb_artifacts}"

# --- 硬件/框架特定配置 ---
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"

# --- 业务配置 ---
PROJECT_NAME='${PROJECT_NAME:-verl-continual-ns-grpo-debug}'
EXP_NAME='${EXP_NAME:-continual-code-math-full}'

# --- 数据与模型路径 (核心修改：使用相对路径作为默认值，允许环境变量覆盖) ---

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/base_model}"
TASK1_TRAIN="${TASK1_TRAIN:-${PROJECT_ROOT}/data/task1/train_data.parquet}"
TASK1_VAL="${TASK1_VAL:-${PROJECT_ROOT}/data/task1/val_data.parquet}"
TASK2_TRAIN="${TASK2_TRAIN:-${PROJECT_ROOT}/data/task2/train_data.parquet}"
TASK2_VAL="${TASK2_VAL:-${PROJECT_ROOT}/data/task2/val_data.parquet}"
SEED_JSONL_PATH="${SEED_JSONL_PATH:-${PROJECT_ROOT}/data/seed.jsonl}"

# --- 自定义奖励函数路径  ---

CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${PROJECT_ROOT}/recipe/continual_ns_grpo/custom_score.py}"

# --- 输出/缓存目录 ---
LOCAL_DIR="${LOCAL_DIR:-${PROJECT_ROOT}/output/multi_task_ckpt}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/output/rollouts}"

# ==============================================================================
# 3. 执行训练命令
# ==============================================================================

python -m recipe.continual_ns_grpo.main_continual_ns_grpo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="[['${TASK1_TRAIN}'],['${TASK2_TRAIN}']]" \
    data.val_files="[['${TASK1_VAL}'],['${TASK2_VAL}']]" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    data.max_response_length=9216 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${LOCAL_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.task_epochs=[1,1] \
    trainer.task_steps=[500,500] \
    actor_rollout_ref.replay.seed_jsonl_path="${SEED_JSONL_PATH}" \
    continual_learning.save_task_checkpoints=True \
    continual_learning.eval_all_tasks=True \
    "$@"