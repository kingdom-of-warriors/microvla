import argparse
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
# 添加项目路径
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter

from model.model_vla import MicroVLA, VLACofig, ActionTokenizer
from trainer.model_utils import create_vla_model
from utils import load_stats

def parse_args():
    """参数解析，无需改动"""
    parser = argparse.ArgumentParser(description="VLA Model Evaluation Script")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained VLA model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="./model", help="Path to the tokenizer")
    parser.add_argument("--benchmark", type=str, default="libero_object", choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"], help="LIBERO benchmark to evaluate on")
    parser.add_argument("--task_id", type=int, default=0, help="Specific task ID to evaluate (0-9)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--max_steps", type=int, default=350, help="Maximum steps per episode")
    parser.add_argument("--save_videos", action="store_true", help="Save evaluation videos")
    parser.add_argument("--video_dir", type=str, default="./eval_videos", help="Directory to save videos")
    parser.add_argument("--results_dir", type=str, default="./eval_results", help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_vla_model(checkpoint_path, model_config_path, device):
    """
    加载训练好的 VLA 模型，并将action_tokenizer附加到模型实例上。
    """
    print("Loading VLA model...")
    
    # 创建配置
    config = VLACofig(hidden_size=768, num_hidden_layers=16, bins=256)
    config.map_path = 'model/action_token_map_256.json'
    config.stats_path = 'dataset/meta/stats.json'

    tokenizer = AutoTokenizer.from_pretrained(model_config_path)
    actions_meta = load_stats(config.stats_path)['actions']
    
    action_tokenizer = ActionTokenizer(
        tokenizer=tokenizer,
        action_token_map_path=config.map_path,
        min_actions=actions_meta['1st'],
        max_actions=actions_meta['99th'],
        bins=config.bins,
    )
    
    # 在初始化模型时，将 action_tokenizer 传入
    model = MicroVLA(
        params=config,
        vision_model_path="./model/vision_model/clip-vit-base-patch16",
        action_tokenizer=action_tokenizer
    )
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    print("VLA model loaded successfully.")
    return model

def raw_obs_to_tensor_obs(obs, device):
    """将【单个】原始观测转换为模型输入格式（batch_size=1）。"""
    stats = load_stats('dataset/meta/stats.json')
    main_tfs = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=stats['image']['mean'],      # [0.485, 0.456, 0.406]
            std=stats['image']['std']         # [0.229, 0.224, 0.225]
        )
    ])
    
    wrist_tfs = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=stats['wrist_image']['mean'], # [0.512, 0.398, 0.321]
            std=stats['wrist_image']['std']    # [0.201, 0.189, 0.243]
        )
    ])
    
    agentview_img = obs['agentview_image'][::-1].copy()
    agentview_img = Image.fromarray(agentview_img.astype('uint8'))
    agentview_img = main_tfs(agentview_img)

    wrist_img = obs['robot0_eye_in_hand_image'][::-1].copy()
    wrist_img = Image.fromarray(wrist_img.astype('uint8'))
    wrist_img = wrist_tfs(wrist_img)

    # 堆叠成 [2, 3, H, W] 并添加 batch 维度
    pixel_values = torch.stack([agentview_img, wrist_img])
    
    return pixel_values.unsqueeze(0).to(device)

def evaluate_task(model: MicroVLA, task, args):
    print(f"Evaluating task: {task.language}")
    
    env_args = {
        "bddl_file_name": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        "camera_heights": 256,
        "camera_widths": 256,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)
    init_states_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path)
    
    num_success = 0
    total_steps = 0
    video_base_dir = os.path.join(args.video_dir, f"task_{args.task_id}")

    for episode in tqdm(range(args.num_episodes), desc=f"Task {args.task_id} Episodes"):
        episode_video_dir = os.path.join(video_base_dir, f"episode_{episode}")
        
        with VideoWriter(episode_video_dir, args.save_videos) as video_writer:
            obs = env.reset()
            
            init_state_idx = episode % len(init_states)
            init_state = init_states[init_state_idx]
            obs = env.set_init_state(init_state)

            for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))

            if args.save_videos:
                video_writer.append_obs(obs, done=False, idx=0, camera_name="agentview_image")

            for step in range(args.max_steps):
                pixel_values = raw_obs_to_tensor_obs(obs, args.device)
                actions_batch = model.predict_action_kv_cache(pixel_values, task.language)
                action = actions_batch[0]
                obs, reward, done, info = env.step(action)
                total_steps += 1
                
                if args.save_videos:
                    video_writer.append_obs(obs, done, idx=0, camera_name="agentview_image")

                if done:
                    num_success += 1
                    break
        
        current_rate = num_success / (episode + 1)
        tqdm.write(f"Episode {episode + 1}/{args.num_episodes} finished. Current success rate: {current_rate:.2f}")

    env.close()
    
    success_rate = num_success / args.num_episodes
    return {
        'task_id': args.task_id,
        'task_name': task.language,
        'success_rate': success_rate,
        'num_success': num_success,
        'num_episodes': args.num_episodes,
        'avg_steps': total_steps / args.num_episodes if args.num_episodes > 0 else 0
    }


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True); os.makedirs(args.video_dir, exist_ok=True)

    config = VLACofig(map_path='model/action_token_map_256.json', stats_path='dataset/meta/stats.json')
    model = create_vla_model(
        config=config,
        action_stats=load_stats(config.stats_path)['actions'],
        rank=0,
        tokenizer_path=args.tokenizer_path,
        ckpt_path=args.ckpt_path
    )[0].to(args.device)
    
    benchmark_map = {"libero_10": "LIBERO_10", "libero_spatial": "LIBERO_SPATIAL", "libero_object": "LIBERO_OBJECT", "libero_goal": "LIBERO_GOAL"}
    benchmark = get_benchmark(benchmark_map[args.benchmark])(0)
    task = benchmark.get_task(args.task_id)
    print(f"Starting evaluation on {args.benchmark} task {args.task_id}")
    with Timer() as timer:
        results = evaluate_task(model, task, args)
    
    results['evaluation_time'] = timer.get_elapsed_time()
    results_file = os.path.join(args.results_dir, f"vla_eval_{args.benchmark}_task{args.task_id}_seed{args.seed}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n{'='*50}\nEvaluation Results:")
    print(f"Task: {results['task_name']}")
    print(f"Success Rate: {results['success_rate']:.3f}")
    print(f"Success: {results['num_success']}/{results['num_episodes']}")
    print(f"Evaluation Time: {results['evaluation_time']:.2f}s")
    print(f"Results saved to: {results_file}\n{'='*50}")


if __name__ == "__main__":
    main()