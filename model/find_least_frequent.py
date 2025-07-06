import json
from collections import Counter
from transformers import AutoTokenizer
import os
from tqdm import tqdm

# --- 配置区 ---
MODEL_PATH = 'MiniMind2-V'
# 更改为文件路径列表
DATASET_PATHS = [
    'sft_data/sft_data_zh.jsonl',
    'sft_data/sft_data_en.jsonl'
]
NUM_TOKENS_TO_REPLACE = 128
OUTPUT_MAPPING_FILE = 'action_token_map.json'
# --- 配置区结束 ---

def process_multiple_files_robustly():
    # --- 1. 加载 Tokenizer ---
    print(f"步骤 1: 正在从 '{MODEL_PATH}' 加载分词器...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型路径 '{MODEL_PATH}'。")
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    print(f"分词器加载成功。词汇表大小: {vocab_size}")

    # --- 2. 健壮地加载所有文件到内存 ---
    all_data = []
    total_line_count = 0
    total_error_lines = 0

    # 遍历文件列表中的每一个路径
    for file_path in DATASET_PATHS:
        print(f"\n步骤 2: 准备从 '{file_path}' 加载文件到内存...")
        if not os.path.exists(file_path):
            print(f"警告: 找不到数据集文件 '{file_path}'，将跳过此文件。")
            continue

        try:
            # 首先获取当前文件总行数以便tqdm显示进度
            with open(file_path, 'r', encoding='utf-8') as f_count:
                num_lines = sum(1 for _ in f_count)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # 使用tqdm显示当前文件的处理进度
                for line in tqdm(f, total=num_lines, desc=f"Reading {os.path.basename(file_path)}"):
                    total_line_count += 1
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        total_error_lines += 1
                        continue
            
            print(f"文件 '{file_path}' 加载完成。")

        except Exception as e:
            print(f"处理文件 '{file_path}' 时发生严重错误: {e}")
            # 即使一个文件出错，也可以选择继续处理下一个文件
            continue
            
    print("\n----------------------------------------------------")
    print(f"所有文件加载完成。")
    print(f"总共扫描 {total_line_count} 行，成功加载 {len(all_data)} 条记录，跳过 {total_error_lines} 条损坏的行。")
    print("----------------------------------------------------")

    # --- 3. 统计词频 ---
    print("\n步骤 3: 正在遍历内存中的合并数据集并统计词频...")
    token_counts = Counter()
    for item in tqdm(all_data, desc="Processing combined dataset"):
        texts_to_process = []
        for key in ["instruction", "input", "output"]:
            if item.get(key): texts_to_process.append(item[key])
        if item.get("history"):
            for turn in item["history"]:
                if isinstance(turn, list): texts_to_process.extend(map(str, turn))
        for text in texts_to_process:
            if text:
                ids = tokenizer.encode(text, add_special_tokens=False)
                token_counts.update(ids)
    print("词频统计完成。")

    # --- 4 & 5 & 6: 找出低频词元，创建映射并验证 (与之前相同) ---
    print(f"\n步骤 4: 正在找出频率最低的 {NUM_TOKENS_TO_REPLACE} 个词元...")
    all_token_ids = vocab.values()
    freqs_for_all_tokens = [(token_id, token_counts.get(token_id, 0)) for token_id in all_token_ids]
    sorted_freqs = sorted(freqs_for_all_tokens, key=lambda item: item[1])
    least_frequent_ids = [item[0] for item in sorted_freqs[:NUM_TOKENS_TO_REPLACE]]

    print(f"\n步骤 5: 正在创建并保存映射文件...")
    action_to_token_id_map = {i: token_id for i, token_id in enumerate(least_frequent_ids)}
    with open(OUTPUT_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(action_to_token_id_map, f, indent=4)
    print(f"映射已成功保存到 '{OUTPUT_MAPPING_FILE}'。")

    print("\n--- 步骤 6: 验证信息 ---")
    print(f"找到的 {NUM_TOKENS_TO_REPLACE} 个最低频词元 (前10个示例):")
    for i in range(min(128, NUM_TOKENS_TO_REPLACE)):
        token_id = least_frequent_ids[i]
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        freq = token_counts.get(token_id, 0)
        print(f"  Token: '{token_str}' (ID: {token_id}) | 语料库中出现频率: {freq}")

if __name__ == "__main__":
    process_multiple_files_robustly()