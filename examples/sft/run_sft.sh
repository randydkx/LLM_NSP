#!/bin/bash
# VERL SFT训练启动脚本
# 用法: bash run_sft.sh <模型路径> <训练数据路径> <验证数据路径> <保存路径> [GPU数量] [其他配置...]

set -x

# 检查必需参数
if [ "$#" -lt 4 ]; then
    echo "用法: run_sft.sh <模型路径> <训练数据路径> <验证数据路径> <保存路径> [GPU数量] [其他配置...]"
    echo ""
    echo "示例:"
    echo "  bash run_sft.sh Qwen/Qwen2.5-0.5B-Instruct \\"
    echo "    ~/data/gsm8k/train.parquet \\"
    echo "    ~/data/gsm8k/test.parquet \\"
    echo "    ~/checkpoints/sft \\"
    echo "    8 \\"
    echo "    trainer.total_epochs=4 \\"
    echo "    optim.lr=1e-4"
    echo ""
    echo "参数说明:"
    echo "  模型路径: HuggingFace模型名称或本地路径"
    echo "  训练数据路径: parquet格式的训练数据文件"
    echo "  验证数据路径: parquet格式的验证数据文件"
    echo "  保存路径: checkpoint保存目录"
    echo "  GPU数量: (可选) 使用的GPU数量，默认8"
    exit 1
fi

# 基础参数
MODEL_PATH=$1
TRAIN_DATA=$2
VAL_DATA=$3
SAVE_PATH=$4
NUM_GPUS=${5:-8}

# 移除前5个参数，剩余的作为额外配置
shift 5

# 设置实验名称（从模型路径提取）
MODEL_NAME=$(basename "$MODEL_PATH")
EXPERIMENT_NAME="sft-${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"

# ===== 选择训练后端 =====
# 支持三种后端:
# 1. fsdp: FSDP后端，使用verl.trainer.fsdp_sft_trainer (默认)
# 2. engine: Engine后端(支持vLLM/SGLang)，使用verl.trainer.sft_trainer
# 3. ray: Ray后端，使用verl.trainer.sft_trainer_ray

BACKEND=${BACKEND:-fsdp}

if [ "$BACKEND" = "fsdp" ]; then
    ENTRYPOINT="-m verl.trainer.fsdp_sft_trainer"
    echo "使用FSDP后端进行训练"
elif [ "$BACKEND" = "engine" ]; then
    ENTRYPOINT="-m verl.trainer.sft_trainer"
    echo "使用Engine后端进行训练"
elif [ "$BACKEND" = "ray" ]; then
    ENTRYPOINT="-m verl.trainer.sft_trainer_ray"
    echo "使用Ray后端进行训练"
else
    echo "错误: 不支持的后端 $BACKEND，请使用 fsdp/engine/ray"
    exit 1
fi

# ===== 启动训练 =====
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    $ENTRYPOINT \
    model.partial_pretrain=$MODEL_PATH \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=verl-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    data.micro_batch_size_per_gpu=128 \
    data.train_batch_size=1024 \
    data.max_length=16384 \
    optim.lr=2e-4 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.model_dtype=bfloat16 \
    $@

# ===== 常用配置说明 =====
# 
# 数据配置:
#   data.prompt_key=question                    # 单轮对话时的prompt字段名
#   data.response_key=answer                    # 单轮对话时的response字段名
#   data.prompt_dict_keys=['question']          # 从嵌套字典中提取prompt
#   +data.response_dict_keys=['answer']         # 从嵌套字典中提取response
#   data.multiturn.enable=true                  # 启用多轮对话模式
#   data.multiturn.messages_key=messages        # 多轮对话的messages字段
#   data.max_length=2048                        # 最大序列长度
#   data.train_batch_size=256                   # 全局batch size
#   data.micro_batch_size_per_gpu=4             # 每个GPU的micro batch size
#
# 优化器配置:
#   optim.lr=1e-5                               # 学习率
#   optim.weight_decay=0.01                     # 权重衰减
#   optim.lr_warmup_steps_ratio=0.1             # warmup步数比例
#   optim.clip_grad=1.0                         # 梯度裁剪
#   optim.lr_scheduler=cosine                   # 学习率调度器
#
# 模型配置:
#   model.fsdp_config.model_dtype=bfloat16      # 模型精度 (fp32/bfloat16/fp16)
#   model.enable_gradient_checkpointing=true    # 启用梯度检查点
#   model.lora_rank=32                          # 启用LoRA (设置rank>0)
#   model.lora_alpha=16                         # LoRA alpha参数
#   model.target_modules=all-linear             # LoRA目标模块
#   model.use_liger=true                        # 启用LigerKernel加速
#   model.trust_remote_code=true                # 信任远程代码
#
# 序列并行配置:
#   ulysses_sequence_parallel_size=2            # Ulysses序列并行大小
#   use_remove_padding=true                     # 启用去除padding优化
#
# 训练器配置:
#   trainer.total_epochs=3                      # 训练轮数
#   trainer.total_training_steps=1000           # 总训练步数（优先于epochs）
#   trainer.save_freq=500                       # checkpoint保存频率
#   trainer.test_freq=500                       # 验证频率
#   trainer.logger='["console","wandb"]'        # 日志记录器
#   trainer.resume_mode=auto                    # 恢复模式 (auto/disable/resume_path)
#   trainer.resume_from_path=/path/to/ckpt      # 从指定路径恢复
#   trainer.max_ckpt_to_keep=5                  # 最多保留的checkpoint数量
#
# 使用示例:
#
# 1. 基础SFT训练:
#   bash run_sft.sh Qwen/Qwen2.5-0.5B-Instruct \
#     ~/data/train.parquet ~/data/val.parquet ~/checkpoints/sft 8
#
# 2. 使用LoRA训练:
#   bash run_sft.sh Qwen/Qwen2.5-7B-Instruct \
#     ~/data/train.parquet ~/data/val.parquet ~/checkpoints/sft-lora 4 \
#     model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear
#
# 3. 长序列训练（使用序列并行）:
#   bash run_sft.sh Qwen/Qwen2.5-7B-Instruct \
#     ~/data/train.parquet ~/data/val.parquet ~/checkpoints/sft-long 8 \
#     data.max_length=8192 ulysses_sequence_parallel_size=2 use_remove_padding=true
#
# 4. 多轮对话训练:
#   bash run_sft.sh Qwen/Qwen2.5-7B-Instruct \
#     ~/data/train.parquet ~/data/val.parquet ~/checkpoints/sft-multiturn 8 \
#     data.multiturn.enable=true data.multiturn.messages_key=messages
#
# 5. 使用WandB记录和更高学习率:
#   bash run_sft.sh Qwen/Qwen2.5-0.5B-Instruct \
#     ~/data/train.parquet ~/data/val.parquet ~/checkpoints/sft 8 \
#     trainer.logger='["console","wandb"]' optim.lr=1e-4 trainer.total_epochs=5
#
# 6. 使用Engine后端（支持vLLM/SGLang）:
#   BACKEND=engine bash run_sft.sh Qwen/Qwen2.5-0.5B-Instruct \
#     ~/data/train.parquet ~/data/val.parquet ~/checkpoints/sft 4
