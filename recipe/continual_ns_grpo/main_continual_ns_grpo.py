"""
Continual Learning NS-GRPO Training with multiple datasets.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import (
    TaskRunner as MainTaskRunner,
)
from verl.trainer.main_ppo import (
    create_rl_dataset,
    create_rl_sampler,
)
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available

from .continual_ray_trainer import ContinualPPOTrainer


@hydra.main(config_path="config", config_name="continual_ns_grpo_trainer", version_base=None)
def main(config):
    """Main entry point for Continual NS-GRPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    run_ppo(config)


# Define a function to run the continual learning PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed continual learning PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner(MainTaskRunner):
    def add_actor_rollout_worker(self, config):
        """Add NSP actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # use new model engine implementation
        if use_legacy_worker_impl == "disable":
            raise NotImplementedError("New model engine not supported for NSP continual learning yet")

        # Always use async worker since sync mode is deprecated
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            # Use custom NSP worker instead of standard worker
            from recipe.continual_ns_grpo.nsp_fsdp_worker import NSPActorRolloutRefWorker

            actor_rollout_cls = NSPActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            raise NotImplementedError("Megatron backend not supported for NSP continual learning")
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def run(self, config):
        """Execute the main continual learning PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets (as a list for continual learning), and reward functions,
        then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the continual learning PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # ========== Continual Learning: Create datasets as lists ==========
        # Expect config.data.train_files and config.data.val_files to be lists of lists
        # e.g., train_files = [[task1_file1, task1_file2], [task2_file1], ...]
        
        train_dataset_list = []
        val_dataset_list = []
        train_files_list = list(config.data.train_files)
        val_files_list = list(config.data.val_files)
        # Check if train_files is a list of lists (continual learning mode)
        if isinstance(train_files_list, list) and len(config.data.train_files) > 0:
            # Continual learning mode: multiple tasks
            num_tasks = len(config.data.train_files)
            print(f"Continual learning mode detected: {num_tasks} tasks")
            
            # Update num_tasks in config
            try:
                OmegaConf.set_struct(config, True)
                with open_dict(config):
                    config.data.num_tasks = num_tasks
                print(f"Updated config.data.num_tasks = {num_tasks}")
            except Exception as e:
                print(f"Warning: Could not set num_tasks in config. Error: {e}")
            
            for task_idx, task_train_files in enumerate(config.data.train_files):
                print(f"Creating dataset for task {task_idx + 1}/{num_tasks}")
                train_dataset = create_rl_dataset(
                    task_train_files,
                    config.data,
                    tokenizer,
                    processor,
                    is_train=True,
                    max_samples=config.data.get("train_max_samples", -1),
                )
                train_dataset_list.append(train_dataset)
        
        # Similarly for validation datasets
        if isinstance(val_files_list, list) and len(config.data.val_files) > 0:
            # Continual learning mode: multiple validation tasks
            for task_idx, task_val_files in enumerate(config.data.val_files):
                val_dataset = create_rl_dataset(
                    task_val_files,
                    config.data,
                    tokenizer,
                    processor,
                    is_train=False,
                    max_samples=config.data.get("val_max_samples", -1),
                )
                val_dataset_list.append(val_dataset)
        # Create sampler list
        train_sampler_list = [create_rl_sampler(config.data, train_dataset) for train_dataset in train_dataset_list]

        # Initialize the Continual PPO trainer with dataset lists
        trainer = ContinualPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset_list,  # Pass as list
            val_dataset=val_dataset_list,      # Pass as list
            collate_fn=collate_fn,
            train_sampler=train_sampler_list,  # Pass as list
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the continual learning training process.
        trainer.fit()


if __name__ == "__main__":
    main()
