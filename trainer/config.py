import argparse
from dataclasses import dataclass, field
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_vla import VLACofig

@dataclass
class TrainingConfig:
    """训练超参数配置"""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    use_wandb: bool = False
    
    # VLA模型相关配置
    vla_config: VLACofig = field(default_factory=lambda: VLACofig(
        hidden_size=768, 
        num_hidden_layers=16, 
        max_seq_len=8192, 
        bins=256,
        map_path='model/action_token_map_256_new.json'
    ))
    
    # 数据集相关配置
    data_split_ratio: float = 0.9
    max_seq_length: int = 256

def parse_args() -> TrainingConfig:
    """解析命令行参数并覆盖默认配置"""
    parser = argparse.ArgumentParser(description='MicroVLA DDP Training')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable wandb logging')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    config = TrainingConfig(use_wandb=args.use_wandb)
    
    # 如果命令行提供了参数，则覆盖默认值
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
        
    return config