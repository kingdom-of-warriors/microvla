from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from utils import load_stats, load_task_info

# 1. 加载数据集
ds = load_dataset("physical-intelligence/libero")['train']


# 3. 创建自定义PyTorch Dataset类
class LiberoDataset(Dataset):
    def __init__(self, ds, task_file_path='dataset/meta/tasks.jsonl', stats_path='dataset/meta/stats.json'):
        self.dataset = ds
        self.stats_path = stats_path
        self.task_info = load_task_info(task_file_path) # 加载任务信息
        self.stats = load_stats(self.stats_path)
        self.main_tfs = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.stats['image']['mean'],      # [0.485, 0.456, 0.406]
                std=self.stats['image']['std']         # [0.229, 0.224, 0.225]
            )
        ])
        self.wrist_tfs = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.stats['wrist_image']['mean'], # [0.512, 0.398, 0.321]
                std=self.stats['wrist_image']['std']    # [0.201, 0.189, 0.243]
            )
        ])


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 处理图像
        image = self.main_tfs(sample['image'])
        wrist_image = self.wrist_tfs(sample['wrist_image'])

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
    
    # def compute_and_save_action_quantiles(self, quantiles=[1, 99]):
    #     """
    #     计算动作数据的分位数并保存到stats.json文件
    #     quantiles: 要计算的分位数列表，例如 [1, 99]
    #     stats_path: stats.json文件路径
    #     """
    #     stats_path = self.stats_path
    #     total_samples = len(self.dataset)
    #     all_actions = []
        
    #     for i in tqdm(range(total_samples), desc="Loading actions"):
    #         sample = self.dataset[i]
    #         actions = np.array(sample['actions'])
    #         all_actions.append(actions)
        
    #     all_actions = np.array(all_actions)
    #     print(f"Action data shape: {all_actions.shape}")
        
    #     # 计算每个维度的分位数
    #     action_quantiles = {}
    #     for q in quantiles: 
    #         action_quantiles[f'{q}th_percentile'] = np.percentile(all_actions, q, axis=0).tolist()
        
    #     # 计算其他统计信息
    #     action_quantiles['mean'] = np.mean(all_actions, axis=0).tolist()
    #     action_quantiles['std'] = np.std(all_actions, axis=0).tolist()
    #     action_quantiles['min'] = np.min(all_actions, axis=0).tolist()
    #     action_quantiles['max'] = np.max(all_actions, axis=0).tolist()
        
    #     # 读取现有的stats.json
    #     with open(stats_path, 'r', encoding='utf-8') as f:
    #         stats_data = json.load(f)

    #     # 更新actions部分
    #     if 'actions' not in stats_data: stats_data['actions'] = {}
    #     stats_data['actions'].update(action_quantiles)
    #     with open(stats_path, 'w', encoding='utf-8') as f:
    #         json.dump(stats_data, f, indent=4, ensure_ascii=False)
        
    #     print(f"Action quantiles saved to {stats_path}")
        
    #     # 打印结果
    #     print("\nAction Statistics (7 dimensions):")
    #     print("=" * 60)
    #     for i in range(7):  # 假设有7个动作维度
    #         print(f"Dim {i}:")
    #         if '1st_percentile' in action_quantiles:
    #             print(f"  1st percentile:  {action_quantiles['1st_percentile'][i]:.6f}")
    #         if '99th_percentile' in action_quantiles:
    #             print(f"  99th percentile: {action_quantiles['99th_percentile'][i]:.6f}")
    #         print(f"  Mean:            {action_quantiles['mean'][i]:.6f}")
    #         print(f"  Std:             {action_quantiles['std'][i]:.6f}")
    #         print(f"  Min:             {action_quantiles['min'][i]:.6f}")
    #         print(f"  Max:             {action_quantiles['max'][i]:.6f}")
    #         print("-" * 40)
        
    #     return action_quantiles