import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import List

# 从项目文件中导入必要的类
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import LiberoDataset
from model.model_vla import MicroVLA, VLACofig, ActionTokenizer

def init_trained_vlm(config: VLACofig, action_1st: List[float], action_99th: List[float]):
    """
    加载一个已经训练好的VLM模型作为VLA的基础。
    """
    print("Initializing MicroVLA model...")
    tokenizer = AutoTokenizer.from_pretrained('./model')
    # 设置tokenizer
    action_tokenizer = ActionTokenizer(tokenizer=tokenizer,
                                       action_token_map_path=config.map_path,
                                       min_actions=action_1st,
                                       max_actions=action_99th,
                                       bins=config.bins)
    transformers_model_path = 'MiniMind2-V'
    model = MicroVLA(config, vision_model_path="./model/vision_model/clip-vit-base-patch16", action_tokenizer=action_tokenizer)
    
    # 加载预训练的VLM权重
    print("Loading pretrained VLM weights...")
    temp_vlm_model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    vlm_state_dict = temp_vlm_model.state_dict()
    model.load_state_dict({k: v for k, v in vlm_state_dict.items()}, strict=False)
    
    # 清理临时模型
    del vlm_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("VLM weights loaded successfully.")
    print("Model initialized.")
    return model, action_tokenizer

def split_dataset(dataset: LiberoDataset, train_ratio=0.8, val_ratio=0.2):
    """
    按episode_index划分数据集，确保同一轨迹的数据在同一个split中。
    """
    print("Splitting dataset by episode...")
    # 获取所有唯一的episode_index
    unique_episodes = np.unique(dataset['episode_index'])
    
    # 划分episodes
    train_episodes, val_episodes = train_test_split(
        unique_episodes, 
        train_size=train_ratio, 
        random_state=42
    )
    
    train_episodes = set(train_episodes)
    val_episodes = set(val_episodes)
    
    # 根据划分好的episodes过滤数据集
    train_indices = [i for i, ep_idx in enumerate(dataset['episode_index']) if ep_idx in train_episodes]
    val_indices = [i for i, ep_idx in enumerate(dataset['episode_index']) if ep_idx in val_episodes]
    
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    print(f"Split complete. Train episodes: {len(train_episodes)}, Val episodes: {len(val_episodes)}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def collate_fn(batch, action_tokenizer: ActionTokenizer, max_length: int = 256):
    pixel_values_list = []
    input_ids_list = []
    attention_mask_list = []

    # 特殊token ID
    bos_id = action_tokenizer.bos_token_id
    eos_id = action_tokenizer.eos_token_id
    pad_id = action_tokenizer.pad_token_id
    for sample in batch:
        # 1. 处理图像：将主图像和手腕图像堆叠
        pixel_values = torch.stack([sample['image'], sample['wrist_image']])
        pixel_values_list.append(pixel_values)  
        instruction_ids = action_tokenizer.tokenizer.encode(sample['task_description'], add_special_tokens=False)
        action_ids = action_tokenizer.encode(sample['actions'].numpy())
        
        # 3. 组合成完整的输入序列: <bos> + instruction + actions + <eos>
        input_ids = [bos_id] + instruction_ids + list(action_ids) + [eos_id]
        
        # 截断或填充
        seq_len = len(input_ids)
        if seq_len > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * seq_len + [0] * (max_length - seq_len)
            input_ids += [pad_id] * (max_length - seq_len)
            
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))

    batched_pixel_values = torch.stack(pixel_values_list)
    batched_input_ids = torch.stack(input_ids_list)
    batched_attention_mask = torch.stack(attention_mask_list)

    return {
        'pixel_values': batched_pixel_values,
        'input_ids': batched_input_ids,
        'attention_mask': batched_attention_mask,
    }

def compute_action_metrics(outputs, batch, action_tokenizer: ActionTokenizer, num_vision_patches=196):
    """
    计算动作预测准确率
    """
    action_preds = outputs['logits'][:, 2 * num_vision_patches + 1:, ].argmax(dim=2)
    action_gt = batch['input_ids'][:, 1:].to(action_preds.device)
    
    # 创建mask来标识真正的动作token位置
    action_token_ids = set(action_tokenizer.action_to_token_id.values())
    mask = torch.zeros_like(action_gt, dtype=torch.bool)
    for token_id in action_token_ids:
        mask |= (action_gt == token_id)
    if mask.sum() == 0: return None, None
    
    # 计算准确率
    correct_preds = (action_preds == action_gt) & mask
    action_accuracy = correct_preds.sum().float() / mask.sum().float()
    
    return action_accuracy

def train_vla():
    # --- 1. 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = VLACofig(hidden_size=768, num_hidden_layers=16, max_seq_len=8192, bins=256)
    config.map_path = 'model/action_token_map_256_new.json' # 指定动作token映射文件

    # 训练超参数
    epochs = 5
    batch_size = 8
    learning_rate = 1e-5
    
    # --- 2. 准备数据集 ---
    dataset = load_dataset("physical-intelligence/libero")['train']
    train_ds, val_ds = split_dataset(dataset)

    train_dataset = LiberoDataset(ds=train_ds, 
                                  task_file_path='dataset/meta/tasks_zh.jsonl',
                                  stats_path='dataset/meta/stats.json')
    val_dataset = LiberoDataset(ds=val_ds, 
                                task_file_path='dataset/meta/tasks_zh.jsonl',
                                stats_path='dataset/meta/stats.json')
    # 1和99分位数
    action_1st  = train_dataset.stats['actions']['1st']
    action_99th = train_dataset.stats['actions']['99th']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, action_tokenizer),
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, action_tokenizer),
        num_workers=0
    )

    # --- 3. 初始化模型和分词器 ---
    model, action_tokenizer = init_trained_vlm(config=config, action_1st=action_1st, action_99th=action_99th)
    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    frozen_params = total_params - trainable_params
    
    print(f"\n=== 模型参数统计 ===")
    print(f"可训练参数量：{trainable_params:,} ({trainable_params / 1e6:.3f}M)")
    print(f"冻结参数量：{frozen_params:,} ({frozen_params / 1e6:.3f}M)")
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.3f}M)")
    print(f"可训练参数比例：{trainable_params / total_params * 100:.2f}%")
    print("=" * 30)

    # --- 4. 设置优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 5. 训练循环 ---
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_action_acc = 0
        valid_batches_train = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                use_text_token=False
            )
            
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # 计算动作指标
            action_acc = compute_action_metrics(outputs, batch, action_tokenizer)
            if action_acc is not None:
                total_train_action_acc += action_acc.item()
                valid_batches_train += 1
                
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': action_acc.item(),
                })
            else:
                train_pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_action_acc / valid_batches_train if valid_batches_train > 0 else 0
        
        print(f"Epoch {epoch+1} - Training Metrics:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Action Accuracy: {avg_train_acc:.4f}")

        # 验证
        model.eval()
        total_val_loss = 0
        total_val_action_acc = 0
        valid_batches_val = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = outputs['loss']
                total_val_loss += loss.item()
                
                # 计算动作指标
                action_acc = compute_action_metrics(outputs, batch, action_tokenizer)
                if action_acc is not None:
                    total_val_action_acc += action_acc.item()
                    valid_batches_val += 1
                    
                    val_pbar.set_postfix({
                        'val_loss': loss.item(),
                        'val_acc': action_acc.item(),
                    })
                else:
                    val_pbar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_action_acc / valid_batches_val if valid_batches_val > 0 else 0
        
        print(f"Epoch {epoch+1} - Validation Metrics:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  Action Accuracy: {avg_val_acc:.4f}")
        print("-" * 50)

        # 保存模型检查点
        torch.save(model.state_dict(), f"vla_zh_checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_vla()
