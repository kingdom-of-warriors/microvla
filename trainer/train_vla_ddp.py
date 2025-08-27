# [DDP 修改] 导入DDP所需库
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from typing import List

# 从项目文件中导入必要的类
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import LiberoDataset
from model.model_vla import MicroVLA, VLACofig, ActionTokenizer

IGNORE_INDEX = -100

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MicroVLA DDP Training')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Enable wandb logging (default: False)')
    return parser.parse_args()

def setup_ddp():
    """初始化DDP进程组"""
    # torchrun会自动设置这些环境变量
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    print(f"DDP setup on rank {rank}, device {torch.cuda.current_device()}")
    return rank, local_rank, world_size

def init_trained_vlm(config: VLACofig, action_1st: List[float], action_99th: List[float], rank: int):
    if rank == 0: print("Initializing MicroVLA model...")
    tokenizer = AutoTokenizer.from_pretrained('./model')
    action_tokenizer = ActionTokenizer(tokenizer=tokenizer,
                                       action_token_map_path=config.map_path,
                                       min_actions=action_1st,
                                       max_actions=action_99th,
                                       bins=config.bins)
    transformers_model_path = 'MiniMind2-V'
    model = MicroVLA(config, vision_model_path="./model/vision_model/clip-vit-base-patch16", action_tokenizer=action_tokenizer)

    if rank == 0: print("Loading pretrained VLM weights...")
    temp_vlm_model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    model.load_state_dict({k: v for k, v in temp_vlm_model.state_dict().items()}, strict=False)
    
    del temp_vlm_model; torch.cuda.empty_cache()

    if rank == 0: print("Model initialized.")
    return model, action_tokenizer

def split_dataset(dataset: LiberoDataset, train_ratio=0.8, rank=0):
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

def collate_fn(batch, action_tokenizer: ActionTokenizer, max_length: int = 256):
    pixel_values_list, input_ids_list, attention_mask_list = [], [], []
    bos_id, eos_id, pad_id = action_tokenizer.bos_token_id, action_tokenizer.eos_token_id, action_tokenizer.pad_token_id
    for sample in batch:
        pixel_values = torch.stack([sample['image'], sample['wrist_image']])
        pixel_values_list.append(pixel_values)
        instruction_ids = action_tokenizer.tokenizer.encode(sample['task_description'], add_special_tokens=False)
        action_ids = action_tokenizer.encode(sample['actions'].numpy())
        input_ids = [bos_id] + instruction_ids + list(action_ids) + [eos_id]
        seq_len = len(input_ids)
        if seq_len > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * seq_len + [0] * (max_length - seq_len)
            input_ids += [pad_id] * (max_length - seq_len)
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
    return {'pixel_values': torch.stack(pixel_values_list), 'input_ids': torch.stack(input_ids_list), 'attention_mask': torch.stack(attention_mask_list)}

def compute_action_metrics(outputs, batch, action_tokenizer: ActionTokenizer, num_vision_patches=196):
    action_preds = outputs['logits'][:, 2 * num_vision_patches + 1:, ].argmax(dim=2)
    action_gt = batch['input_ids'][:, 1:].to(action_preds.device)
    action_token_ids = set(action_tokenizer.action_to_token_id.values())
    mask = torch.zeros_like(action_gt, dtype=torch.bool)
    for token_id in action_token_ids:
        mask |= (action_gt == token_id)
    if mask.sum() == 0: return None
    correct_preds = (action_preds == action_gt) & mask
    return correct_preds.sum().float() / mask.sum().float()

def detect_loss(outputs, shift_labels, action_tokenizer: ActionTokenizer, rank):
    predicted_tokens = outputs["logits"].argmax(dim=-1)[:, :-1]  # [bs, length-1]
    # 找到有效标签的位置（非-100的位置就是动作位置）
    valid_mask = (shift_labels != IGNORE_INDEX)  # [bs, length]
    action_predictions = predicted_tokens[valid_mask]  # [num_action_tokens]
    action_token_ids = set(action_tokenizer.action_to_token_id.values())
    valid_action_count = sum(token.item() in action_token_ids for token in action_predictions)
    
    # if rank == 0: print(f"预测了 {len(action_predictions)} 个动作token，其中 {valid_action_count} 个是有效动作token")

def train_vla():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    # [WANDB 添加] 只在主进程初始化wandb
    if rank == 0 and args.use_wandb:
        wandb.init(
            project="microvla_debug",
            name=f"vla_training_ddp_ws{world_size}",
            config={
                "epochs": 5,
                "batch_size_per_gpu": 8,
                "total_batch_size": 8 * world_size,
                "learning_rate": 1e-5,
                "world_size": world_size,
                "hidden_size": 768,
                "num_hidden_layers": 16,
                "max_seq_len": 8192,
                "bins": 256
            }
        )
    elif rank == 0: print('wandb disabled by --use_wandb flag')

    config = VLACofig(hidden_size=768, num_hidden_layers=16, max_seq_len=8192, bins=256)
    config.map_path = 'model/action_token_map_256_new.json'

    epochs = 5
    batch_size = 8
    learning_rate = 1e-5
    
    # --- 2. 准备数据集 ---
    if rank == 0: print("Loading and splitting dataset...")
    dataset = load_dataset("physical-intelligence/libero")['train']
    train_ds, val_ds = split_dataset(dataset, 0.8, rank)

    train_dataset = LiberoDataset(ds=train_ds, task_file_path='dataset/meta/tasks.jsonl', stats_path='dataset/meta/stats.json')
    val_dataset = LiberoDataset(ds=val_ds, task_file_path='dataset/meta/tasks.jsonl', stats_path='dataset/meta/stats.json')
    action_1st, action_99th = train_dataset.stats['actions']['1st'], train_dataset.stats['actions']['99th']

    # [DDP 修改] 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, action_tokenizer), num_workers=0, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, action_tokenizer), num_workers=0, sampler=val_sampler)

    # --- 3. 初始化并封装模型 ---
    model, action_tokenizer = init_trained_vlm(config=config, action_1st=action_1st, action_99th=action_99th, rank=rank)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params per GPU: {trainable_params / 1e6:.3f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 5. 训练循环 ---
    for epoch in range(epochs):
        # 设置sampler的epoch，确保每个epoch的shuffle不同
        train_sampler.set_epoch(epoch)
        
        model.train()
        total_train_loss = 0
        total_train_action_acc = 0
        valid_batches_train = 0
        
        # [DDP 修改] 只在主进程显示tqdm进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", disable=(rank != 0))
        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs, shift_labels = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, use_text_token=False)
            # detect_loss(outputs, shift_labels, action_tokenizer, rank)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # [DDP 修改] 在每个进程上计算指标，然后在验证环节进行同步
            total_train_loss += loss.item()
            action_acc = compute_action_metrics(outputs, batch, action_tokenizer)
            if action_acc is not None:
                total_train_action_acc += action_acc.item()
                valid_batches_train += 1
            
            if rank == 0:
                train_pbar.set_postfix({'loss': loss.item()})
                # 记录每个batch的训练指标
                if args.use_wandb and batch_idx % 50 == 0:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_action_acc": action_acc.item() if action_acc is not None else 0,
                        "epoch": epoch + 1,
                        "step": epoch * len(train_loader) + batch_idx
                    })
        train_metrics = torch.tensor([total_train_loss, total_train_action_acc, valid_batches_train, len(train_loader)]).to(device)
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)

        # 验证 (validation)
        model.eval()
        val_loss_sum, val_acc_sum, val_batches_sum = 0.0, 0.0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", disable=(rank != 0))
        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                
                val_loss_sum += outputs['loss'].item()
                
                action_acc = compute_action_metrics(outputs, batch, action_tokenizer)
                if action_acc is not None:
                    val_acc_sum += action_acc.item()
                    val_batches_sum += 1

        # [DDP 修改] 同步所有进程的验证指标
        val_metrics = torch.tensor([val_loss_sum, val_acc_sum, val_batches_sum, len(val_loader)]).to(device)
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
        # [DDP 修改] 只在主进程计算并打印最终的平均指标
        if rank == 0:
            total_train_loss_all, total_train_acc_all, total_train_batches_all, total_train_loader_len = train_metrics.tolist()
            avg_train_loss = total_train_loss_all / total_train_loader_len
            avg_train_acc = total_train_acc_all / total_train_batches_all if total_train_batches_all > 0 else 0
            
            total_val_loss, total_val_acc, total_val_batches, total_val_loader_len = val_metrics.tolist()
            avg_val_loss = total_val_loss / total_val_loader_len
            avg_val_acc = total_val_acc / total_val_batches if total_val_batches > 0 else 0
            
            # 记录epoch级别的指标
            if args.use_wandb: wandb.log({
                "train/epoch_loss": avg_train_loss,
                "train/epoch_action_acc": avg_train_acc,
                "val/epoch_loss": avg_val_loss,
                "val/epoch_action_acc": avg_val_acc,
                "epoch": epoch + 1
            })

            print(f"\n--- Epoch {epoch+1} Training Metrics ---")
            print(f"  Avg Train Loss: {avg_train_loss:.4f}")
            print(f"  Avg Train Action Accuracy: {avg_train_acc:.4f}")
            print(f"\n--- Epoch {epoch+1} Validation Metrics ---")
            print(f"  Avg Val Loss: {avg_val_loss:.4f}")
            print(f"  Avg Val Action Accuracy: {avg_val_acc:.4f}")
            print("-" * 50)

            checkpoint_path = f"vla_checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # 结束wandb运行
    if rank == 0 and args.use_wandb: wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    train_vla()