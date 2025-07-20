from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image

import json

# 1. 加载数据集
ds = load_dataset("physical-intelligence/libero")['train']

# 2. 加载任务信息
def load_task_info(task_file_path):
    """加载任务信息，返回task_index到task描述的映射"""
    task_info = {}
    with open(task_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                task_data = json.loads(line.strip())
                task_info[task_data['task_index']] = task_data['task']
    return task_info

# 3. 创建自定义PyTorch Dataset类
class LiberoDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, task_file_path='dataset/meta/tasks.jsonl'):
        self.dataset = hf_dataset
        self.transform = transform
        # 加载任务信息
        self.task_info = load_task_info(task_file_path)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 处理图像
        image = self.transform(sample['image'])
        wrist_image = self.transform(sample['wrist_image'])

        # 处理其他数据
        state = torch.tensor(sample['state'], dtype=torch.float32)
        actions = torch.tensor(sample['actions'], dtype=torch.float32)
        # timestamp = sample['timestamp']
        # frame_index = sample['frame_index']
        # episode_index = sample['episode_index']
        task_index = sample['task_index']
        
        # 根据task_index获取任务描述
        task_description = self.task_info.get(task_index, f"Unknown task {task_index}")
        
        return {
            'image': image,
            'wrist_image': wrist_image,
            'state': state,
            'actions': actions,
            'task_description': task_description,
            # 'task_index': task_index,
            # 'episode_index': episode_index,
            # 'frame_index': frame_index,
            # 'timestamp': timestamp,
            # 'index': sample['index']
        }

# 4. 测试任务信息加载
if __name__ == "__main__":
    # 测试任务信息加载
    task_info = load_task_info('dataset/meta/tasks.jsonl')
    print("加载的任务信息示例：")
    for i in range(5):
        print(f"Task {i}: {task_info.get(i, 'Not found')}")
    
    # 测试数据集
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LiberoDataset(ds, transform=transform)
    
    # 查看数据示例
    sample = dataset[1000]
    print(f"\n数据示例 (index 1000):")
    print(f"Task description: {sample['task_description']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"State shape: {sample['state'].shape}")