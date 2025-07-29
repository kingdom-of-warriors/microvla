import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- DDP 相关导入 ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 从项目文件中导入必要的类
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import LiberoDataset
from model.model_vla import MicroVLA, VLACofig, ActionTokenizer

# --- DDP 初始化函数 ---
def setup_ddp():
    """初始化DDP环境"""
    dist.init_process_group(backend="nccl")
    # local_rank 决定了当前进程使用哪块GPU
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

# --- 模型和数据集函数 (基本不变) ---
def init_trained_vlm(config: VLACofig):
    """
    加载一个已经训练好的VLM模型作为VLA的基础。
    """
    print("Initializing MicroVLA model...")
    tokenizer = AutoTokenizer.from_pretrained('./model')
    transformers_model_path = 'MiniMind2-V'
    model = MicroVLA(config, vision_model_path="./model/vision_model/clip-vit-base-patch16")
    
    # 加载预训练的VLM权重
    print("Loading pretrained VLM weights...")
    temp_vlm_model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    vlm_state_dict = temp_vlm_model.state_dict()
    model.load_state_dict({k: v for k, v in vlm_state_dict.items()}, strict=False)
    
    # 清理临时模型
    del vlm_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 设置tokenizer
    action_tokenizer = ActionTokenizer(tokenizer=tokenizer,
                                       action_token_map_path=config.map_path,
                                       bins=config.bins)
    
    print("VLM weights loaded successfully.")
    print("Model initialized.")
    return model, action_tokenizer
    
def split_dataset(dataset: LiberoDataset, train_ratio=0.8, val_ratio=0.2):
    # ... (此函数内容无需修改) ...
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
    bos_id = action_tokenizer.tokenizer.bos_token_id
    eos_id = action_tokenizer.tokenizer.eos_token_id
    pad_id = action_tokenizer.tokenizer.pad_token_id
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
    if mask.sum() == 0: return None
    
    # 计算准确率
    correct_preds = (action_preds == action_gt) & mask
    action_accuracy = correct_preds.sum().float() / mask.sum().float()
    
    return action_accuracy

def train_vla_ddp(local_rank):
    """ DDP版本的训练主函数 """
    
    # --- 1. 配置 ---
    # DDP中，device就是local_rank
    device = torch.device(f"cuda:{local_rank}")
    # 通过dist.get_rank()获取全局rank
    global_rank = dist.get_rank()
    print(f"Running DDP on rank {global_rank}, device {device}.")
    
    config = VLACofig(hidden_size=768, num_hidden_layers=16, max_seq_len=8192, bins=256)
    config.map_path = 'model/action_token_map_256_new.json'

    epochs = 5
    batch_size = 8
    learning_rate = 1e-5
    
    # --- 2. 初始化模型和分词器 ---
    # 只在主进程(rank 0)中打印信息和加载
    if global_rank == 0:
        print("=== Initializing Model and Tokenizer (on rank 0) ===")
    model, action_tokenizer = init_trained_vlm(config)
    model = model.to(device)
    
    # *** DDP核心：用DDP包装模型 ***
    model = DDP(model, device_ids=[local_rank])
    
    # 只在主进程打印参数统计
    if global_rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = total_params - trainable_params
        
        print(f"\n=== 模型参数统计 ===")
        print(f"可训练参数量：{trainable_params:,} ({trainable_params / 1e6:.3f}M)")
        print(f"冻结参数量：{frozen_params:,} ({frozen_params / 1e6:.3f}M)")
        print(f"总参数量：{total_params:,} ({total_params / 1e6:.3f}M)")
        print(f"可训练参数比例：{trainable_params / total_params * 100:.2f}%")
        print("=" * 30)

    # --- 3. 准备数据集 ---
    # 只在主进程下载和划分数据集，然后所有进程同步
    if global_rank == 0:
        print("Preparing dataset (on rank 0)...")
        dataset = load_dataset("physical-intelligence/libero")['train']
        train_ds, val_ds = split_dataset(dataset)
        
        # 创建LiberoDataset实例
        train_dataset = LiberoDataset(ds=train_ds, task_file_path='dataset/meta/tasks_zh.jsonl')
        val_dataset = LiberoDataset(ds=val_ds, task_file_path='dataset/meta/tasks_zh.jsonl')
    dist.barrier()
    
    if global_rank != 0:
        # 重新加载一遍以确保所有进程都有数据
        dataset = load_dataset("physical-intelligence/libero")['train']
        train_ds, val_ds = split_dataset(dataset)
        train_dataset = LiberoDataset(ds=train_ds, task_file_path='dataset/meta/tasks_zh.jsonl')
        val_dataset = LiberoDataset(ds=val_ds, task_file_path='dataset/meta/tasks_zh.jsonl')

    # *** DDP核心：使用DistributedSampler ***
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, action_tokenizer),
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler, # 使用sampler
        collate_fn=lambda b: collate_fn(b, action_tokenizer),
        num_workers=2
    )

    # --- 4. 设置优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 5. 训练循环 ---
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        
        model.train()
        total_train_loss = 0
        total_train_action_acc = 0
        valid_batches_train = 0
        
        # 只在主进程显示tqdm进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", disable=(global_rank != 0))
        for batch in train_pbar:
            optimizer.zero_grad()
            
            # 数据移动到当前进程对应的GPU
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = outputs['loss']
            loss.backward() # DDP会自动同步梯度
            optimizer.step()
            
            total_train_loss += loss.item()
            
            action_acc = compute_action_metrics(outputs, batch, action_tokenizer)
            if action_acc is not None:
                total_train_action_acc += action_acc.item()
                valid_batches_train += 1
                
                # 更新主进程的进度条
                if global_rank == 0:
                    train_pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': action_acc.item(),
                    })
            elif global_rank == 0:
                train_pbar.set_postfix({'loss': loss.item()})
        
        # --- 多卡指标同步 ---
        total_train_loss_tensor = torch.tensor(total_train_loss).to(device)
        dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
        
        # 只在主进程计算并打印
        if global_rank == 0:
            avg_train_loss = total_train_loss_tensor.item() / (len(train_loader) * dist.get_world_size())
            avg_train_acc = total_train_action_acc / valid_batches_train if valid_batches_train > 0 else 0
            
            print(f"\nEpoch {epoch+1} - Training Metrics (Aggregated):")
            print(f"  Loss: {avg_train_loss:.4f}")
            print(f"  Action Accuracy (rank 0 avg): {avg_train_acc:.4f}")

        # *** DDP核心：只在主进程保存模型 ***
        if global_rank == 0:
            # 保存时需要获取原始模型，而不是DDP包装后的模型
            torch.save(model.module.state_dict(), f"vla_zh_checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch+1}")
        
        # 等待所有进程完成，再进入下一个epoch
        dist.barrier()


if __name__ == "__main__":
    local_rank = setup_ddp()
    train_vla_ddp(local_rank)
    cleanup_ddp()