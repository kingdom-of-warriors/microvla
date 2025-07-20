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
NUM_TOKENS_TO_REPLACE = 256
OUTPUT_MAPPING_FILE = 'action_token_map_256_new.json'
# --- 配置区结束 ---

def get_special_token_ids(tokenizer):
    """获取所有特殊token的ID"""
    special_token_ids = set()
    
    # 获取常见的特殊token
    special_tokens = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id, 
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else None,
        tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else None,
        tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else None,
    ]
    
    # 过滤掉None值并添加到集合
    for token_id in special_tokens:
        if token_id is not None:
            special_token_ids.add(token_id)
    
    # 获取所有special_tokens_map中的token
    if hasattr(tokenizer, 'special_tokens_map'):
        for token_name, token_value in tokenizer.special_tokens_map.items():
            if isinstance(token_value, str):
                token_id = tokenizer.convert_tokens_to_ids(token_value)
                if token_id is not None:
                    special_token_ids.add(token_id)
    
    # 获取additional_special_tokens
    if hasattr(tokenizer, 'additional_special_tokens'):
        for token in tokenizer.additional_special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None:
                special_token_ids.add(token_id)
    
    # 获取all_special_ids属性（如果存在）
    if hasattr(tokenizer, 'all_special_ids'):
        special_token_ids.update(tokenizer.all_special_ids)
    
    return special_token_ids

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

    # --- 1.5. 获取特殊token ID ---
    special_token_ids = get_special_token_ids(tokenizer)
    print(f"检测到 {len(special_token_ids)} 个特殊token，将从候选列表中排除:")
    for token_id in sorted(special_token_ids):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"  特殊Token: '{token_str}' (ID: {token_id})")

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

    # --- 4. 找出低频词元，排除特殊token ---
    print(f"\n步骤 4: 正在找出频率最低的 {NUM_TOKENS_TO_REPLACE} 个词元（排除特殊token）...")
    all_token_ids = vocab.values()
    
    # 过滤掉特殊token
    candidate_token_ids = [token_id for token_id in all_token_ids if token_id not in special_token_ids]
    print(f"排除 {len(special_token_ids)} 个特殊token后，候选token数量: {len(candidate_token_ids)}")
    
    if len(candidate_token_ids) < NUM_TOKENS_TO_REPLACE:
        print(f"错误: 候选token数量 ({len(candidate_token_ids)}) 少于所需数量 ({NUM_TOKENS_TO_REPLACE})")
        return
    
    # 计算频率并排序
    freqs_for_candidate_tokens = [(token_id, token_counts.get(token_id, 0)) for token_id in candidate_token_ids]
    sorted_freqs = sorted(freqs_for_candidate_tokens, key=lambda item: item[1])
    least_frequent_ids = [item[0] for item in sorted_freqs[:NUM_TOKENS_TO_REPLACE]]

    # --- 5. 创建映射并保存 ---
    print(f"\n步骤 5: 正在创建并保存映射文件...")
    action_to_token_id_map = {i: token_id for i, token_id in enumerate(least_frequent_ids)}
    with open(OUTPUT_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(action_to_token_id_map, f, indent=4)
    print(f"映射已成功保存到 '{OUTPUT_MAPPING_FILE}'。")

    # --- 6. 验证信息 ---
    print("\n--- 步骤 6: 验证信息 ---")
    print(f"找到的 {NUM_TOKENS_TO_REPLACE} 个最低频词元 (前10个示例):")
    for i in range(min(10, NUM_TOKENS_TO_REPLACE)):
        token_id = least_frequent_ids[i]
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        freq = token_counts.get(token_id, 0)
        print(f"  Token: '{token_str}' (ID: {token_id}) | 语料库中出现频率: {freq}")
    
    # 验证没有特殊token被选中
    selected_special_tokens = special_token_ids.intersection(set(least_frequent_ids))
    if selected_special_tokens:
        print(f"\n警告: 发现特殊token被意外选中: {selected_special_tokens}")
    else:
        print(f"\n✓ 验证通过: 没有特殊token被选为动作token")

if __name__ == "__main__":
    process_multiple_files_robustly()