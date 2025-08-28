import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_vla import MicroVLA, VLACofig, ActionTokenizer

def create_vla_model(config: VLACofig, action_stats: dict, rank: int) -> Tuple[MicroVLA, ActionTokenizer]:
    """初始化MicroVLA模型、分词器并加载预训练权重"""
    if rank == 0: print("Initializing MicroVLA model...")
    
    # 1. 加载 actiontokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model')
    action_tokenizer = ActionTokenizer(
        tokenizer=tokenizer,
        action_token_map_path=config.map_path,
        min_actions=action_stats['actions']['1st'],
        max_actions=action_stats['actions']['99th'],
        bins=config.bins
    )
    
    # 2. 初始化模型
    model = MicroVLA(
        config, 
        vision_model_path="./model/vision_model/clip-vit-base-patch16", 
        action_tokenizer=action_tokenizer
    )
    
    if rank == 0: print("Loading pretrained VLM weights...")
    # 3. 加载预训练权重
    with torch.no_grad():
        temp_vlm_model = AutoModelForCausalLM.from_pretrained('MiniMind2-V', trust_remote_code=True)
        model.load_state_dict({k: v for k, v in temp_vlm_model.state_dict().items()}, strict=False)
    del temp_vlm_model; torch.cuda.empty_cache()
    if rank == 0: print("Model initialized.")

    return model, action_tokenizer