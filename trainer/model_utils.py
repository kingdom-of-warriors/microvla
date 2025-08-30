import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_vla import MicroVLA, VLACofig, ActionTokenizer

def create_vla_model(config: VLACofig, 
                     action_stats: dict, 
                     rank: int,
                     tokenizer_path: str, 
                     ckpt_path: str) -> Tuple[MicroVLA, ActionTokenizer]:
    if rank == 0: print("Initializing MicroVLA model...")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    action_tokenizer = ActionTokenizer(
        tokenizer=tokenizer,
        action_token_map_path=config.map_path,
        min_actions=action_stats['1st'],
        max_actions=action_stats['99th'],
        bins=config.bins
    )
    
    model = MicroVLA(
        config, 
        vision_model_path="./model/vision_model/clip-vit-base-patch16", 
        action_tokenizer=action_tokenizer
    )

    # 3. 根据 ckpt_path 类型选择加载方式
    if ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt'):
        if rank == 0: print(f"Loading checkpoint from {ckpt_path}")
        with torch.no_grad():
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
    else:
        if rank == 0: print("Loading pretrained VLM weights...")
        with torch.no_grad():
            temp_vlm_model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True)
            model.load_state_dict({k: v for k, v in temp_vlm_model.state_dict().items()}, strict=False)
    
    if rank == 0: print("Model initialized.")

    return model, action_tokenizer