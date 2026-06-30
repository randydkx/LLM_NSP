#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERL SFT训练数据准备示例脚本

这个脚本展示了如何准备不同格式的训练数据并保存为parquet格式
"""

import pandas as pd
import argparse
import os


def create_single_turn_simple_format():
    """
    格式1: 单轮对话 - 简单格式
    适用场景: 最简单的问答对
    
    配置:
        data.prompt_key=question
        data.response_key=answer
    """
    data = [
        {
            "question": "什么是机器学习？",
            "answer": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出决策，而无需明确编程。"
        },
        {
            "question": "Python中如何创建列表？",
            "answer": "在Python中，你可以使用方括号[]创建列表，例如：my_list = [1, 2, 3, 4, 5]"
        },
        {
            "question": "解释一下什么是深度学习？",
            "answer": "深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的复杂模式和表示。"
        }
    ]
    
    df = pd.DataFrame(data)
    return df, "single_turn_simple"


def create_single_turn_nested_format():
    """
    格式2: 单轮对话 - 嵌套格式
    适用场景: 数据包含额外信息（metadata）
    
    配置:
        data.prompt_key=extra_info
        data.response_key=extra_info
        data.prompt_dict_keys=['question']
        +data.response_dict_keys=['answer']
    """
    data = [
        {
            "extra_info": {
                "question": "计算: 5 + 3 × 2",
                "answer": "11",
                "difficulty": "easy",
                "category": "math"
            }
        },
        {
            "extra_info": {
                "question": "一个苹果3元，买5个多少钱？",
                "answer": "15元",
                "difficulty": "easy",
                "category": "math"
            }
        }
    ]
    
    df = pd.DataFrame(data)
    return df, "single_turn_nested"


def create_multiturn_format():
    """
    格式3: 多轮对话格式
    适用场景: 对话历史、多轮交互
    
    配置:
        data.multiturn.enable=true
        data.multiturn.messages_key=messages
    """
    data = [
        {
            "messages": [
                {"role": "user", "content": "你好，我想学习Python"},
                {"role": "assistant", "content": "你好！很高兴帮助你学习Python。你想从哪里开始呢？"},
                {"role": "user", "content": "从基础语法开始吧"},
                {"role": "assistant", "content": "好的！让我们从变量和数据类型开始。Python有几种基本数据类型..."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "什么是列表推导式？"},
                {"role": "assistant", "content": "列表推导式是Python中创建列表的简洁方式。例如：[x*2 for x in range(5)]"},
                {"role": "user", "content": "能给个更复杂的例子吗？"},
                {"role": "assistant", "content": "当然！比如筛选偶数并平方：[x**2 for x in range(10) if x % 2 == 0]"}
            ]
        }
    ]
    
    df = pd.DataFrame(data)
    return df, "multiturn"


def create_code_instruction_format():
    """
    格式4: 代码指令格式
    适用场景: 代码生成任务
    """
    data = [
        {
            "instruction": "写一个Python函数，计算斐波那契数列的第n项",
            "output": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
        },
        {
            "instruction": "实现一个快速排序算法",
            "output": """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)"""
        }
    ]
    
    df = pd.DataFrame(data)
    return df, "code_instruction"


def create_math_cot_format():
    """
    格式5: 数学推理格式（Chain of Thought）
    适用场景: 需要推理步骤的数学问题
    """
    data = [
        {
            "problem": "小明有15个苹果，他给了小红5个，又给了小华3个，他还剩多少个苹果？",
            "solution": "让我们一步步计算：\n1. 小明开始有15个苹果\n2. 给了小红5个后，剩余：15 - 5 = 10个\n3. 又给了小华3个后，剩余：10 - 3 = 7个\n\n因此，小明还剩7个苹果。"
        },
        {
            "problem": "一个长方形的长是8cm，宽是5cm，求它的周长和面积",
            "solution": "让我们分别计算周长和面积：\n\n周长计算：\n周长 = 2 × (长 + 宽)\n周长 = 2 × (8 + 5)\n周长 = 2 × 13 = 26cm\n\n面积计算：\n面积 = 长 × 宽\n面积 = 8 × 5 = 40cm²\n\n答案：周长为26cm，面积为40cm²"
        }
    ]
    
    df = pd.DataFrame(data)
    return df, "math_cot"


def main():
    parser = argparse.ArgumentParser(description="生成不同格式的SFT训练数据示例")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/sft_examples",
        help="输出目录"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["all", "simple", "nested", "multiturn", "code", "math"],
        default="all",
        help="要生成的数据格式"
    )
    
    args = parser.parse_args()
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    formats = {
        "simple": create_single_turn_simple_format,
        "nested": create_single_turn_nested_format,
        "multiturn": create_multiturn_format,
        "code": create_code_instruction_format,
        "math": create_math_cot_format,
    }
    
    if args.format == "all":
        selected_formats = formats
    else:
        selected_formats = {args.format: formats[args.format]}
    
    print(f"📁 输出目录: {output_dir}\n")
    
    for format_name, format_fn in selected_formats.items():
        df, file_prefix = format_fn()
        
        # 分割训练集和验证集（这里简单地用前80%做训练，后20%做验证）
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size] if train_size > 0 else df
        val_df = df[train_size:] if len(df) > train_size else df[:1]  # 至少保留一条验证数据
        
        train_path = os.path.join(output_dir, f"{file_prefix}_train.parquet")
        val_path = os.path.join(output_dir, f"{file_prefix}_val.parquet")
        
        train_df.to_parquet(train_path)
        val_df.to_parquet(val_path)
        
        print(f"✅ 生成 {format_name} 格式:")
        print(f"   训练集: {train_path} ({len(train_df)} 条)")
        print(f"   验证集: {val_path} ({len(val_df)} 条)")
        print(f"   列名: {list(df.columns)}")
        print()
    
    print("\n" + "="*60)
    print("📚 使用说明:")
    print("="*60)
    
    print("\n1️⃣  简单格式 (single_turn_simple):")
    print("   bash examples/sft/run_sft.sh \\")
    print("     Qwen/Qwen2.5-0.5B-Instruct \\")
    print(f"     {output_dir}/single_turn_simple_train.parquet \\")
    print(f"     {output_dir}/single_turn_simple_val.parquet \\")
    print("     ~/checkpoints/sft-simple 4 \\")
    print("     data.prompt_key=question data.response_key=answer")
    
    print("\n2️⃣  嵌套格式 (single_turn_nested):")
    print("   bash examples/sft/run_sft.sh \\")
    print("     Qwen/Qwen2.5-0.5B-Instruct \\")
    print(f"     {output_dir}/single_turn_nested_train.parquet \\")
    print(f"     {output_dir}/single_turn_nested_val.parquet \\")
    print("     ~/checkpoints/sft-nested 4 \\")
    print("     data.prompt_key=extra_info data.response_key=extra_info \\")
    print("     data.prompt_dict_keys=['question'] +data.response_dict_keys=['answer']")
    
    print("\n3️⃣  多轮对话格式 (multiturn):")
    print("   bash examples/sft/run_sft.sh \\")
    print("     Qwen/Qwen2.5-0.5B-Instruct \\")
    print(f"     {output_dir}/multiturn_train.parquet \\")
    print(f"     {output_dir}/multiturn_val.parquet \\")
    print("     ~/checkpoints/sft-multiturn 4 \\")
    print("     data.multiturn.enable=true data.multiturn.messages_key=messages")
    
    print("\n4️⃣  代码指令格式 (code_instruction):")
    print("   bash examples/sft/run_sft.sh \\")
    print("     Qwen/Qwen2.5-0.5B-Instruct \\")
    print(f"     {output_dir}/code_instruction_train.parquet \\")
    print(f"     {output_dir}/code_instruction_val.parquet \\")
    print("     ~/checkpoints/sft-code 4 \\")
    print("     data.prompt_key=instruction data.response_key=output")
    
    print("\n5️⃣  数学推理格式 (math_cot):")
    print("   bash examples/sft/run_sft.sh \\")
    print("     Qwen/Qwen2.5-0.5B-Instruct \\")
    print(f"     {output_dir}/math_cot_train.parquet \\")
    print(f"     {output_dir}/math_cot_val.parquet \\")
    print("     ~/checkpoints/sft-math 4 \\")
    print("     data.prompt_key=problem data.response_key=solution")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
