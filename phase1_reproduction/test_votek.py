#!/usr/bin/env python
"""
简化版 vote-k 测试脚本
用于测试 vote-k 算法与传统 random 方法的对比
"""

import os
import sys
import argparse
import numpy as np
from get_task import get_task
from two_steps import selective_annotation, prompt_retrieval
from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel

def setup_args(output_dir, method='random', annotation_size=100, debug=False):
    """设置命令行参数"""
    args = argparse.Namespace(
        task_name='dbpedia_14',
        model_cache_dir='models',
        data_cache_dir='datasets',
        output_dir=output_dir,
        model_name='gpt2-xl',
        annotation_size=annotation_size,
        batch_size=8 if not debug else 1,
        seed=0,
        debug=debug,
        selective_annotation_method=method,
        prompt_retrieval_method='similar',
        model_key=None,
        embedding_model='sentence-transformers/paraphrase-mpnet-base-v2'
    )
    return args

def run_test(method='random', annotation_size=100, debug=False):
    """运行单个测试"""
    output_dir = f'outputs/{method}'
    os.makedirs(output_dir, exist_ok=True)

    args = setup_args(output_dir, method, annotation_size, debug)

    print(f"运行 {method} 方法...")
    print(f"  输出目录: {output_dir}")
    print(f"  注释大小: {annotation_size}")
    print(f"  调试模式: {debug}")

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    train_examples, eval_examples, train_text, eval_text, format_example, label_map = get_task(args)
    print(f"  训练示例: {len(train_examples)}")
    print(f"  评估示例: {len(eval_examples)}")

    # 2. 计算嵌入
    print("\n[2/4] 计算嵌入...")
    from utils import calculate_sentence_transformer_embedding
    train_embeds = calculate_sentence_transformer_embedding(train_text, args)
    eval_embeds = calculate_sentence_transformer_embedding(eval_text, args)
    print(f"  训练嵌入: {train_embeds.shape}")
    print(f"  评估嵌入: {eval_embeds.shape}")

    # 3. 选择性注释
    print("\n[3/4] 选择性注释...")
    selected_indices = selective_annotation(
        embeddings=train_embeds,
        train_examples=train_examples,
        return_string=False,
        format_example=format_example,
        maximum_input_len=1000,
        label_map=label_map,
        single_context_example_len=250,
        inference_model=None,
        inference_data_module=None,
        tokenizer_gpt=None,
        args=args
    )
    print(f"  选择了 {len(selected_indices)} 个示例")

    # 4. 结果保存
    import json
    result_file = os.path.join(output_dir, 'selected_indices.json')
    with open(result_file, 'w') as f:
        json.dump(selected_indices, f, indent=2)
    print(f"\n[4/4] 结果已保存到: {result_file}")

    return selected_indices

def compare_methods():
    """对比 vote-k 和 random 方法"""
    print("=" * 60)
    print("Vote-K vs Random 对比测试")
    print("=" * 60)

    # 运行 random 方法
    print("\n### 运行 Random 方法 ###\n")
    random_indices = run_test('random', annotation_size=100, debug=False)

    # 运行 vote-k 方法
    print("\n### 运行 Vote-K 方法 ###\n")
    votek_indices = run_test('fast_votek', annotation_size=100, debug=False)

    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"Random 选择: {len(random_indices)} 个示例")
    print(f"Vote-K 选择: {len(votek_indices)} 个示例")
    print(f"重叠示例: {len(set(random_indices) & set(votek_indices))} 个")
    print(f"差异率: {(1 - len(set(random_indices) & set(votek_indices)) / len(random_indices)) * 100:.1f}%")

    return random_indices, votek_indices

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Vote-K 算法测试脚本')
    parser.add_argument('--method', type=str, default='compare',
                        choices=['random', 'votek', 'fast_votek', 'compare'],
                        help='运行方法: random, votek, fast_votek, 或 compare (对比两者)')
    parser.add_argument('--annotation_size', type=int, default=100,
                        help='注释大小')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式 (使用更小的数据集)')

    args = parser.parse_args()

    if args.method == 'compare':
        compare_methods()
    else:
        run_test(args.method, args.annotation_size, args.debug)
