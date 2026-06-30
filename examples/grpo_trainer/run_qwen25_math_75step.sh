set -x


dapo_train_path=/mnt/shared-storage-user/safevl-share/hw/math_parquet/train.parquet
aime_test_path=/mnt/shared-storage-user/safevl-share/hw/math_parquet/test.parquet
model_path=/mnt/shared-storage-user/safevl-share/hw/math_ckpt20
max_response_length=16384

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
custom_score_path="${script_dir}/custom_math_score.py"
custom_reward_manager_path="${script_dir}/custom_math_reward_manager.py"

train_files="['$dapo_train_path']"
test_files="['$aime_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path="$custom_score_path" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.accuracy_weight=1.0 \
    +custom_reward_function.reward_kwargs.format_weight=0.1 \
    +custom_reward_function.reward_kwargs.format_complete_reward=1.0 \
    +custom_reward_function.reward_kwargs.format_missing_reward=-1.0 \
    reward_manager.source=importlib \
    reward_manager.name=CustomMathRewardManager \
    reward_manager.module.path="$custom_reward_manager_path" \
    reward_model.use_reward_loop=False \
    reward_model.launch_reward_fn_async=False \
    +reward_model.reward_kwargs.length_reward_cfg.enable=True \
    +reward_model.reward_kwargs.length_reward_cfg.weight=1.0 \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.warmup_style="constant" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_math' \
    trainer.experiment_name='mixture_data_plus_length_penalty' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=5 $@
