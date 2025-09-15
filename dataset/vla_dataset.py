import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from PIL import Image
from utils import load_task_info, load_stats
import numpy as np
from scipy.spatial.transform import Rotation

class LiberoDataset(Dataset):
    def __init__(self, ds, task_file_path='dataset/meta/tasks.jsonl', stats_path='dataset/meta/stats.json', 
                 augment=False):
        """
        Args:
            ds: 数据集
            task_file_path: 任务信息文件路径
            stats_path: 统计信息文件路径
            no_augment: 是否禁用数据增强，True表示不使用数据增强
        """
        self.dataset = ds
        self.stats_path = stats_path
        self.task_info = load_task_info(task_file_path)
        self.stats = load_stats(self.stats_path)
        self.augment = augment
        
        # 根据 no_augment 参数选择不同的图像变换
        if not self.augment:
            self.same_tfs = T.Compose([
                T.Resize((224, 224))
            ])
            self.main_tfs = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=self.stats['image']['mean'],
                    std=self.stats['image']['std']
                )
            ])
            self.wrist_tfs = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=self.stats['wrist_image']['mean'],
                    std=self.stats['wrist_image']['std']
                )
            ])
        else:
            self.same_tfs = T.Compose([
                T.RandomResizedCrop((224, 224), scale=(0.95, 1.0), ratio=(0.95, 1.05))
            ])
            self.main_tfs = T.Compose([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=self.stats['image']['mean'],
                    std=self.stats['image']['std']
                )
            ])
            self.wrist_tfs = T.Compose([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=self.stats['wrist_image']['mean'],
                    std=self.stats['wrist_image']['std']
                )
            ])

        # state 归一化参数
        self.state_mean = torch.tensor(self.stats['state']['mean'], dtype=torch.float32)
        self.state_std = torch.tensor(self.stats['state']['std'], dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        wrist_image = sample['wrist_image']
        
        # 图像处理
        transformed_images = self.same_tfs({'image': image, 'wrist_image': wrist_image})
        final_image = self.main_tfs(transformed_images['image'])
        final_wrist_image = self.wrist_tfs(transformed_images['wrist_image'])
        
        # state 处理
        state = torch.tensor(sample['state'], dtype=torch.float32)
        normalized_state = (state - self.state_mean) / self.state_std
        actions = torch.tensor(sample['actions'], dtype=torch.float32)
        task_index = sample['task_index']
        task_description = self.task_info.get(task_index, f"Unknown task {task_index}")
        
        return {
            'image': final_image,
            'wrist_image': final_wrist_image,
            'state': normalized_state,
            'actions': actions,
            'task_description': task_description,
        }
    

def raw_obs_to_tensor_obs(obs, device):
    """将【单个Libero】原始观测转换为模型输入格式（batch_size=1）。"""
    stats = load_stats('dataset/meta/stats.json')
    main_tfs = T.Compose([
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=stats['image']['mean'],      # [0.485, 0.456, 0.406]
            std=stats['image']['std']         # [0.229, 0.224, 0.225]
        )
    ])
    
    wrist_tfs = T.Compose([
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=stats['wrist_image']['mean'], # [0.512, 0.398, 0.321]
            std=stats['wrist_image']['std']    # [0.201, 0.189, 0.243]
        )
    ])
    
    # 构建8维state： "x", "y", "z", "roll", "pitch", "yaw", "gripper", "-gripper"
    ee_pos = obs['robot0_eef_pos'] 
    ee_quat = obs['robot0_eef_quat']
    # "wxyz" to "xyzw" for scipy rotation
    ee_euler = Rotation.from_quat(np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])).as_euler('xyz', degrees=False)
    gripper_qpos = obs['robot0_gripper_qpos']
    state = np.concatenate((ee_pos, ee_euler, gripper_qpos), axis=0)
    
    # state归一化
    state_mean = torch.tensor(stats['state']['mean'], dtype=torch.float32)
    state_std = torch.tensor(stats['state']['std'], dtype=torch.float32)
    state_tensor = torch.tensor(state, dtype=torch.float32)
    normalized_state = (state_tensor - state_mean) / state_std
    state_values = normalized_state.unsqueeze(0).to(device)  # [1, 8]
    
    # 图像处理
    agentview_img = obs['agentview_image'][::-1].copy()
    agentview_img = Image.fromarray(agentview_img.astype('uint8'))
    agentview_img = main_tfs(agentview_img)
    
    wrist_img = obs['robot0_eye_in_hand_image'][::-1].copy()
    wrist_img = Image.fromarray(wrist_img.astype('uint8'))
    wrist_img = wrist_tfs(wrist_img)
    
    pixel_values = torch.stack([agentview_img, wrist_img])
    pixel_values = pixel_values.unsqueeze(0).to(device)  # [1, 2, 3, 224, 224]
    
    return pixel_values, state_values
    
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