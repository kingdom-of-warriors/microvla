import json

def load_task_info(task_file_path: str):
    """加载任务信息，返回task_index到task描述的映射"""
    task_info = {}
    with open(task_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                task_data = json.loads(line.strip())
                task_info[task_data['task_index']] = task_data['task']
    return task_info

def load_stats(stats_path: str = 'dataset/meta/stats.json'):
    """
    从stats.json文件加载统计信息
    """
    with open(stats_path, 'r', encoding='utf-8') as f: stats = json.load(f)
    image_mean = [stats['image']['mean'][i][0][0] for i in range(3)]
    image_std = [stats['image']['std'][i][0][0] for i in range(3)]
    wrist_mean = [stats['wrist_image']['mean'][i][0][0] for i in range(3)]
    wrist_std = [stats['wrist_image']['std'][i][0][0] for i in range(3)]
    actions_stats = stats['actions']
    return {
        'image': {
            'mean': image_mean,
            'std': image_std
        },
        'wrist_image': {
            'mean': wrist_mean,
            'std': wrist_std
        },
        'actions': {
            '1st': actions_stats['1st_percentile'],
            '99th': actions_stats['99th_percentile']
        }
    }