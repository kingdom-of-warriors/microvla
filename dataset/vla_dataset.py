import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from utils import load_task_info, load_stats

class LiberoDataset(Dataset):
    def __init__(self, ds, task_file_path='dataset/meta/tasks.jsonl', stats_path='dataset/meta/stats.json'):
        self.dataset = ds
        self.stats_path = stats_path
        self.task_info = load_task_info(task_file_path)
        self.stats = load_stats(self.stats_path)

        self.same_tfs = T.Compose([T.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1))])

        self.main_tfs = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=self.stats['image']['mean'],
                std=self.stats['image']['std']
            )
        ])

        self.wrist_tfs = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=self.stats['wrist_image']['mean'],
                std=self.stats['wrist_image']['std']
            )
        ])


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        wrist_image = sample['wrist_image']

        transformed_images = self.same_tfs({'image': image, 'wrist_image': wrist_image})
        final_image = self.main_tfs(transformed_images['image'])
        final_wrist_image = self.wrist_tfs(transformed_images['wrist_image'])
        state = torch.tensor(sample['state'], dtype=torch.float32)
        actions = torch.tensor(sample['actions'], dtype=torch.float32)
        task_index = sample['task_index']
        
        task_description = self.task_info.get(task_index, f"Unknown task {task_index}")
        
        return {
            'image': final_image,
            'wrist_image': final_wrist_image,
            'state': state,
            'actions': actions,
            'task_description': task_description,
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