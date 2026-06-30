# VERL SFT 数据格式说明

## 📋 数据格式总览

VERL 的 SFT 训练支持多种数据格式，所有数据必须保存为 **Parquet 格式**。

## 🎯 支持的数据格式

### 1️⃣ 单轮对话 - 简单格式（最常用）

**数据结构:**
```python
{
    "question": "你的问题",
    "answer": "模型的回答"
}
```

**训练配置:**
```bash
data.prompt_key=question
data.response_key=answer
```

**示例数据:**
```python
data = [
    {
        "question": "什么是机器学习？",
        "answer": "机器学习是人工智能的一个分支..."
    },
    {
        "question": "Python中如何创建列表？",
        "answer": "使用方括号[]创建列表..."
    }
]
```

**使用方法:**
```bash
bash examples/sft/run_sft.sh \
  Qwen/Qwen2.5-0.5B-Instruct \
  ~/data/train.parquet \
  ~/data/val.parquet \
  ~/checkpoints/sft \
  4 \
  data.prompt_key=question \
  data.response_key=answer
```

---

### 2️⃣ 单轮对话 - 嵌套格式

**数据结构:**
```python
{
    "extra_info": {
        "question": "你的问题",
        "answer": "模型的回答",
        "metadata1": "其他信息1",
        "metadata2": "其他信息2"
    }
}
```

**训练配置:**
```bash
data.prompt_key=extra_info
data.response_key=extra_info
data.prompt_dict_keys=['question']
+data.response_dict_keys=['answer']
```

**示例数据:**
```python
data = [
    {
        "extra_info": {
            "question": "5 + 3 × 2 = ?",
            "answer": "11",
            "difficulty": "easy",
            "category": "math",
            "source": "gsm8k"
        }
    }
]
```

**使用方法:**
```bash
bash examples/sft/run_sft.sh \
  Qwen/Qwen2.5-7B-Instruct \
  ~/data/train.parquet \
  ~/data/val.parquet \
  ~/checkpoints/sft \
  8 \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  data.prompt_dict_keys=['question'] \
  +data.response_dict_keys=['answer']
```

---

### 3️⃣ 多轮对话格式

**数据结构:**
```python
{
    "messages": [
        {"role": "user", "content": "第一轮用户消息"},
        {"role": "assistant", "content": "第一轮助手回复"},
        {"role": "user", "content": "第二轮用户消息"},
        {"role": "assistant", "content": "第二轮助手回复"}
    ]
}
```

**训练配置:**
```bash
data.multiturn.enable=true
data.multiturn.messages_key=messages
```

**示例数据:**
```python
data = [
    {
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
            {"role": "user", "content": "介绍一下Python"},
            {"role": "assistant", "content": "Python是一种高级编程语言..."}
        ]
    }
]
```

**使用方法:**
```bash
bash examples/sft/run_sft.sh \
  Qwen/Qwen2.5-7B-Instruct \
  ~/data/multiturn_train.parquet \
  ~/data/multiturn_val.parquet \
  ~/checkpoints/sft-multiturn \
  8 \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages
```

---

### 4️⃣ 代码指令格式

**数据结构:**
```python
{
    "instruction": "编程任务描述",
    "output": "生成的代码"
}
```

**训练配置:**
```bash
data.prompt_key=instruction
data.response_key=output
```

**示例数据:**
```python
data = [
    {
        "instruction": "写一个Python函数计算斐波那契数列",
        "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    }
]
```

---

### 5️⃣ 数学推理格式（Chain of Thought）

**数据结构:**
```python
{
    "problem": "数学问题",
    "solution": "详细的解题步骤"
}
```

**训练配置:**
```bash
data.prompt_key=problem
data.response_key=solution
```

**示例数据:**
```python
data = [
    {
        "problem": "小明有15个苹果，给了小红5个，还剩多少？",
        "solution": "让我们一步步计算：\n1. 小明开始有15个苹果\n2. 给了小红5个\n3. 剩余：15 - 5 = 10个"
    }
]
```

---

## 🛠️ 快速生成示例数据

我们提供了示例脚本来生成不同格式的数据：

```bash
# 生成所有格式的示例数据
python examples/sft/prepare_data_example.py --output_dir ~/data/sft_examples

# 只生成特定格式
python examples/sft/prepare_data_example.py --format simple
python examples/sft/prepare_data_example.py --format multiturn
```

生成后会看到：
```
~/data/sft_examples/
├── single_turn_simple_train.parquet
├── single_turn_simple_val.parquet
├── single_turn_nested_train.parquet
├── single_turn_nested_val.parquet
├── multiturn_train.parquet
├── multiturn_val.parquet
├── code_instruction_train.parquet
├── code_instruction_val.parquet
├── math_cot_train.parquet
└── math_cot_val.parquet
```

---

## 📝 创建自己的数据集

### 方法1: 使用 Pandas（推荐）

```python
import pandas as pd

# 准备你的数据
data = [
    {"question": "问题1", "answer": "答案1"},
    {"question": "问题2", "answer": "答案2"},
    # ... 更多数据
]

# 转换为DataFrame并保存
df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

### 方法2: 使用 HuggingFace Datasets

```python
from datasets import Dataset

# 准备数据
data = {
    "question": ["问题1", "问题2"],
    "answer": ["答案1", "答案2"]
}

# 创建数据集并保存
dataset = Dataset.from_dict(data)
dataset.to_parquet("train.parquet")
```

### 方法3: 从 JSON/JSONL 转换

```python
import pandas as pd
import json

# 从JSONL读取
with open("data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# 转换并保存
df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

---

## ⚙️ 数据配置参数详解

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data.train_files` | 训练数据路径 | 必需 |
| `data.val_files` | 验证数据路径 | 必需 |
| `data.max_length` | 最大序列长度 | 1024 |
| `data.train_batch_size` | 全局batch size | 256 |
| `data.micro_batch_size_per_gpu` | 每GPU的micro batch size | 4 |

### 单轮对话参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `data.prompt_key` | prompt字段名 | `question` |
| `data.response_key` | response字段名 | `answer` |
| `data.prompt_dict_keys` | 从嵌套字典提取prompt | `['question']` |
| `data.response_dict_keys` | 从嵌套字典提取response | `['answer']` |

### 多轮对话参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data.multiturn.enable` | 启用多轮对话 | `false` |
| `data.multiturn.messages_key` | messages字段名 | `messages` |
| `data.multiturn.tools_key` | tools字段名 | `tools` |

---

## 💡 常见问题

### Q1: 我的数据字段名不是 question/answer，怎么办？

使用 `prompt_key` 和 `response_key` 指定你的字段名：
```bash
data.prompt_key=my_input_field \
data.response_key=my_output_field
```

### Q2: 数据在嵌套的字典里，如何提取？

使用 `prompt_dict_keys` 和 `response_dict_keys`：
```bash
data.prompt_key=data \
data.response_key=data \
data.prompt_dict_keys=['input'] \
+data.response_dict_keys=['output']
```

### Q3: 支持多个 parquet 文件吗？

支持！可以传入列表：
```bash
data.train_files='[~/data/part1.parquet,~/data/part2.parquet]'
```

### Q4: 如何限制数据量进行快速测试？

使用 `train_max_samples` 和 `val_max_samples`：
```bash
data.train_max_samples=100 \
data.val_max_samples=10
```

### Q5: 数据太长超过 max_length 怎么办？

设置截断策略：
```bash
data.truncation=right  # 或 left, error
```

---

## 📚 完整示例

### 示例1: GSM8K 数学推理

```bash
# 1. 准备数据
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# 2. 开始训练
bash examples/sft/run_sft.sh \
  Qwen/Qwen2.5-7B-Instruct \
  ~/data/gsm8k/train.parquet \
  ~/data/gsm8k/test.parquet \
  ~/checkpoints/gsm8k-sft \
  8 \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  data.prompt_dict_keys=['question'] \
  +data.response_dict_keys=['answer']
```

### 示例2: 多轮对话

```bash
# 1. 准备数据
python examples/data_preprocess/gsm8k_multiturn_sft.py --local_save_dir ~/data/gsm8k_sft

# 2. 开始训练
bash examples/sft/run_sft.sh \
  Qwen/Qwen2.5-7B-Instruct \
  ~/data/gsm8k_sft/train.parquet \
  ~/data/gsm8k_sft/test.parquet \
  ~/checkpoints/multiturn-sft \
  8 \
  data.multiturn.enable=true
```

### 示例3: 自定义简单数据

```python
# create_my_data.py
import pandas as pd

data = [
    {"input": "介绍Python", "output": "Python是一种高级编程语言..."},
    {"input": "什么是AI", "output": "AI是人工智能的缩写..."}
]

df = pd.DataFrame(data)
df.to_parquet("my_train.parquet")
```

```bash
bash examples/sft/run_sft.sh \
  Qwen/Qwen2.5-0.5B-Instruct \
  my_train.parquet \
  my_train.parquet \
  ~/checkpoints/my-sft \
  4 \
  data.prompt_key=input \
  data.response_key=output
```

---

## 🔗 相关资源

- [数据预处理示例](../data_preprocess/)
- [SFT训练配置说明](../../verl/trainer/config/sft_trainer.yaml)
- [完整文档](../../docs/)
