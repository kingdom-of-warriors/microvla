import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import json
from pathlib import Path
from transformers import AutoTokenizer

# 添加项目路径
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
libero_path = os.path.join(project_root, 'LIBERO')

sys.path.append(project_root)
sys.path.append(libero_path)

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter

from model.model_vla import MicroVLA, VLACofig, ActionTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="VLA Model Evaluation Script")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained VLA model checkpoint")
    parser.add_argument("--model_config_path", type=str, default="./model",
                        help="Path to the model configuration")
    parser.add_argument("--benchmark", type=str, default="libero_10",
                        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
                        help="LIBERO benchmark to evaluate on")
    parser.add_argument("--task_id", type=int, default=0,
                        help="Specific task ID to evaluate (0-9)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run evaluation on")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="Maximum steps per episode")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--video_dir", type=str, default="./eval_videos",
                        help="Directory to save videos")
    parser.add_argument("--results_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

def load_vla_model(checkpoint_path, model_config_path, device):
    """加载训练好的 VLA 模型"""
    print("Loading VLA model...")
    
    # 创建配置
    config = VLACofig(
        hidden_size=768, 
        num_hidden_layers=16, 
        max_seq_len=8192, 
        bins=256
    )
    config.map_path = 'model/action_token_map_256_new.json'
    config.stats_path = 'dataset/meta/stats.json'

    tokenizer = AutoTokenizer.from_pretrained(model_config_path)
    model = MicroVLA(config, vision_model_path="./model/vision_model/clip-vit-base-patch16")
    action_tokenizer = ActionTokenizer(
        tokenizer=tokenizer,
        action_token_map_path=config.map_path,
        bins=config.bins,
    )
    
    # 加载checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    print("VLA model loaded successfully.")
    return model, action_tokenizer

def raw_obs_to_tensor_obs(obs, device):
    """将原始观测转换为模型输入格式"""
    batch_size = len(obs)
    
    images = []
    wrist_images = []
    
    for i in range(batch_size):
        agentview_img = obs[i]['agentview_image']  # (H, W, 3)
        agentview_img = torch.from_numpy(agentview_img).float() / 255.0
        agentview_img = agentview_img.permute(2, 0, 1)  # (3, H, W)
        images.append(agentview_img)
        
        wrist_img = obs[i]['robot0_eye_in_hand_image']  # (H, W, 3)
        wrist_img = torch.from_numpy(wrist_img).float() / 255.0
        wrist_img = wrist_img.permute(2, 0, 1)  # (3, H, W)
        wrist_images.append(wrist_img)
    
    # 堆叠图像 [batch_size, 2, 3, H, W]
    pixel_values = torch.stack([
        torch.stack(images),
        torch.stack(wrist_images)
    ], dim=1)
    
    return pixel_values.to(device)

@torch.no_grad()
def predict_action(model, action_tokenizer, pixel_values, task_description, device):
    """使用 VLA 模型预测动作"""
    instruction_ids = action_tokenizer.tokenizer.encode(
        task_description, 
        add_special_tokens=False, 
        return_tensors='pt'
    ).to(device)
    
    # 添加 BOS token
    bos_id = action_tokenizer.bos_token_id
    input_ids = torch.cat([
        torch.tensor([[bos_id]], device=device),
        instruction_ids
    ], dim=1)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    # 获取下一个token的logits
    next_token_logits = outputs['logits'][:, -1, :]  # [batch_size, vocab_size]
    action_token_ids = list(action_tokenizer.action_to_token_id.values())
    action_mask = torch.zeros_like(next_token_logits)
    action_mask[:, action_token_ids] = 1
    masked_logits = next_token_logits + (action_mask - 1) * 1e9
    predicted_token_ids = torch.argmax(masked_logits, dim=-1)  # [batch_size]
    
    # 解码为动作
    actions = []
    for i in range(predicted_token_ids.shape[0]):
        token_id = predicted_token_ids[i].item()
        if token_id in action_tokenizer.token_id_to_action:
            action_bin = action_tokenizer.token_id_to_action[token_id]
            # 假设这是7维动作的第一维，实际需要根据你的设计调整
            action = np.zeros(7)  # 7维动作
            if action_tokenizer.use_quantile_bins:
                action[0] = action_tokenizer.bin_centers_per_dim[0][action_bin]
            else:
                action[0] = action_tokenizer.bin_centers[action_bin]
            actions.append(action)
        else:
            actions.append(np.zeros(7))
    
    return np.array(actions)

def evaluate_task(model, action_tokenizer, task, args):
    """评估单个任务"""
    print(f"Evaluating task: {task.language}")
    
    env_args = {
        "bddl_file_name": os.path.join(
            get_libero_path("bddl_files"), 
            task.problem_folder, 
            task.bddl_file
        ),
        "camera_heights": 224,
        "camera_widths": 224,
    }
    
    env_num = args.num_episodes
    env = SubprocVectorEnv([
        lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)
    ])
    
    try:
        env.reset()
        env.seed(args.seed)
        
        # 加载初始状态
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices]
        obs = env.set_init_state(init_states_)
        
        # 评估统计
        dones = [False] * env_num
        steps = 0
        num_success = 0
        
        # 预热物理模拟
        for _ in range(5):
            env.step(np.zeros((env_num, 7)))
        
        video_folder = os.path.join(args.video_dir, f"task_{args.task_id}")
        with VideoWriter(video_folder, args.save_videos) as video_writer:
            pbar = tqdm(range(args.max_steps), desc=f"Task {args.task_id}")
            while steps < args.max_steps and not all(dones):
                steps += 1
                pixel_values = raw_obs_to_tensor_obs(obs, args.device)
                actions = predict_action(
                    model, action_tokenizer, pixel_values, 
                    task.language, args.device
                )
                obs, rewards, done, info = env.step(actions)
                
                video_writer.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )
                
                # 更新完成状态
                for k in range(env_num):
                    if not dones[k] and done[k]:
                        dones[k] = True
                        num_success += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'success': f"{num_success}/{env_num}",
                    'rate': f"{num_success/env_num*100:.1f}%"
                })
            
            pbar.close()
    
    finally:
        env.close()
    
    success_rate = num_success / env_num
    return {
        'task_id': args.task_id,
        'task_name': task.language,
        'success_rate': success_rate,
        'num_success': num_success,
        'num_episodes': env_num,
        'steps': steps
    }

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    
    model, action_tokenizer = load_vla_model(
        args.checkpoint_path, 
        args.model_config_path, 
        args.device
    )
    
    # 获取benchmark
    benchmark_map = {
        "libero_10": "LIBERO_10",
        "libero_spatial": "LIBERO_SPATIAL", 
        "libero_object": "LIBERO_OBJECT",
        "libero_goal": "LIBERO_GOAL",
    }
    
    benchmark = get_benchmark(benchmark_map[args.benchmark])(0)
    task = benchmark.get_task(args.task_id)
    print(f"Starting evaluation on {args.benchmark} task {args.task_id}")
    with Timer() as timer:
        results = evaluate_task(model, action_tokenizer, task, args)
    
    results['evaluation_time'] = timer.get_elapsed_time()
    
    results_file = os.path.join(
        args.results_dir, 
        f"vla_eval_{args.benchmark}_task{args.task_id}_seed{args.seed}.json"
    )
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"Task: {results['task_name']}")
    print(f"Success Rate: {results['success_rate']:.3f}")
    print(f"Success: {results['num_success']}/{results['num_episodes']}")
    print(f"Evaluation Time: {results['evaluation_time']:.2f}s")
    print(f"Results saved to: {results_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()