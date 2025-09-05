import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import sys, os
from typing import Tuple, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import LiberoDataset
from model.model_vla import ActionTokenizer

def _split_dataset_by_episode(dataset: Dataset, train_ratio=0.8, rank=0) -> Tuple[Dataset, Dataset]:
    """按episode分割数据集"""
    if rank == 0: print("Splitting dataset by episode...")
    unique_episodes = np.unique(dataset['episode_index'])
    train_episodes, val_episodes = train_test_split(unique_episodes, train_size=train_ratio, random_state=42)
    train_episodes, val_episodes = set(train_episodes), set(val_episodes)
    
    train_indices = [i for i, ep_idx in enumerate(dataset['episode_index']) if ep_idx in train_episodes]
    val_indices = [i for i, ep_idx in enumerate(dataset['episode_index']) if ep_idx in val_episodes]
    
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    if rank == 0: print(f"Split complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset

def _collate_fn(batch, action_tokenizer: ActionTokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    """数据collate函数"""
    pixel_values_list, input_ids_list, state_list, attention_mask_list = [], [], [], []
    bos_id, eos_id, pad_id = action_tokenizer.bos_token_id, action_tokenizer.eos_token_id, action_tokenizer.pad_token_id
    
    for sample in batch:
        # 1. 堆叠图像张量
        pixel_values = torch.stack([sample['image'], sample['wrist_image']]) # (2, C, H, W)
        pixel_values_list.append(pixel_values)
        # 2. 获得文本和动作的输入ID：<bos> + text tokens + action tokens + state tokens + <eos> + <pad>
        instruction_ids = action_tokenizer.tokenizer.encode(sample['task_description'], add_special_tokens=False)
        action_ids = action_tokenizer.encode(sample['actions'].numpy())
        input_ids = [bos_id] + instruction_ids + list(action_ids) + [eos_id]
        # 2.1 截断或填充到max_length
        seq_len = len(input_ids)
        if seq_len > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * seq_len + [0] * (max_length - seq_len)
            input_ids += [pad_id] * (max_length - seq_len)
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        # 3. 注意力掩码堆叠
        attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
        # 4. 状态张量堆叠
        state_list.append(sample['state'])
    return {
        'pixel_values': torch.stack(pixel_values_list), 
        'input_ids': torch.stack(input_ids_list), 
        'attention_mask': torch.stack(attention_mask_list),
        'state_values': torch.stack(state_list)
    }

def create_dataloaders(config, action_tokenizer, rank, world_size) -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    """创建训练和验证数据加载器"""
    if rank == 0: print("Loading and preparing datasets...")
    dataset = load_dataset("physical-intelligence/libero")['train']
    train_ds, val_ds = _split_dataset_by_episode(dataset, config.data_split_ratio, rank) # 按episode分割数据集

    train_dataset = LiberoDataset(ds=train_ds, task_file_path='dataset/meta/tasks.jsonl', stats_path='dataset/meta/stats.json')
    val_dataset = LiberoDataset(ds=val_ds, task_file_path='dataset/meta/tasks.jsonl', stats_path='dataset/meta/stats.json')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    collate_wrapper = lambda b: _collate_fn(b, action_tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_wrapper, num_workers=0, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_wrapper, num_workers=0, sampler=val_sampler)
    
    return train_loader, val_loader, train_sampler