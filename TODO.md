# TODO: 基于 Meta 信息优化数据集加载

## 1. 图像数据优化

### 1.1 使用数据集特定的归一化参数
- [X] 从 `dataset/meta/stats.json` 读取图像统计信息
- [X] 替换 ImageNet 标准参数，使用数据集实际的 mean/std
- [X] 为主图像和手腕图像分别应用不同的归一化参数

```python
# 当前使用 ImageNet 参数
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 改进：使用数据集特定参数
image_mean = [stats['image']['mean'][i] for i in range(3)]
image_std = [stats['image']['std'][i] for i in range(3)]
wrist_mean = [stats['wrist_image']['mean'][i] for i in range(3)]
wrist_std = [stats['wrist_image']['std'][i] for i in range(3)]
```

### 1.2 分别处理不同类型图像
- [X] 为主视角图像和手腕图像创建不同的 transform pipeline
- [X] 考虑两种图像可能有不同的光照、角度特性
- [X] 在 LiberoDataset 中支持多种 transform

## 2. 动作数据优化

### 2.1 处理极端动作值
- [X] 分析动作数据的分布特征（min/max/percentiles）
- [X] 使用 99 百分位数作为剪裁边界，避免极端异常值影响
- [X] 实现动作数据的健壮预处理

```python
# 使用 99 百分位数替代绝对 min/max
action_min_99 = np.percentile(actions, 1)   # 1% 百分位
action_max_99 = np.percentile(actions, 99)  # 99% 百分位
actions_clipped = np.clip(actions, action_min_99, action_max_99)
```

## 3. 实现仅对 action token 的loss计算
- [X] 实现仅对 action token 的loss计算
```python
action_only_labels = text_labels.clone()
action_mask = torch.zeros_like(action_only_labels, dtype=torch.bool)
action_token_ids = set(action_tokenizer.action_to_token_id.values())
for token_id in action_token_ids:
    action_mask |= (action_only_labels == token_id)

# 将非action tokens设为IGNORE_INDEX
action_only_labels[~action_mask] = IGNORE_INDEX

full_labels2 = torch.cat([
    torch.full((batch_size, 1), IGNORE_INDEX, 
            dtype=input_ids.dtype, device=input_ids.device),        # <BOS> token (忽略)
    torch.full((batch_size, num * 196), IGNORE_INDEX, 
            dtype=input_ids.dtype, device=input_ids.device),        # Vision tokens (忽略)
    action_only_labels,                                  # 只有 Action tokens 参与loss，Text tokens被忽略
], dim=1)
```