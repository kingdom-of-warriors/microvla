import os
import numpy as np
import torch
import warnings
from .model_vlm import *
from typing import Optional, Tuple, List
from torch import nn
from transformers import PreTrainedTokenizerBase
from typing import List

warnings.filterwarnings('ignore')

import json
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import List, Union

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

class ActionTokenizer:
    """
    一个动作分词器，使用预先计算好的、非连续的低频词元ID作为动作空间。
    支持每个动作维度使用不同的最小值和最大值。
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        action_token_map_path: str,
        min_actions: List[float] = None,
        max_actions: List[float] = None,
        bins: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_actions = np.array(min_actions)
        self.max_actions = np.array(max_actions)
        self.action_dims = len(min_actions)

        with open(action_token_map_path, 'r') as f:
            action_map_str_keys = json.load(f)
            self.action_to_token_id = {int(k): v for k, v in action_map_str_keys.items()}
            
        self.token_id_to_action = {v: k for k, v in self.action_to_token_id.items()}

        # 为每个动作维度创建bins和bin_centers
        self.bins = []
        self.bin_centers = []
        for i in range(self.action_dims):
            bins_i = np.linspace(self.min_actions[i], self.max_actions[i], self.n_bins)
            bin_centers_i = (bins_i[:-1] + bins_i[1:]) / 2.0
            self.bins.append(bins_i)
            self.bin_centers.append(bin_centers_i)
        
        self.bins = np.array(self.bins)                # shape: [action_dims, n_bins]
        self.bin_centers = np.array(self.bin_centers)  # shape: [action_dims, n_bins-1]

    def encode(self, action: np.ndarray) -> np.ndarray:
        """
        核心编码功能：将连续的动作值编码为对应的【词元ID数组】。
        这是用于模型训练的主要函数。
        """
        action = np.array(action)
        if len(action.shape) == 1: # 单个动作: [action_dims]
            action = action.reshape(1, -1)
            is_single = True
        else: is_single = False  # 批量动作: [batch_size, action_dims]
        
        batch_size, action_dims = action.shape
        token_ids = np.zeros_like(action, dtype=int)
        
        for dim in range(action_dims):
            clipped_action = np.clip(action[:, dim], self.min_actions[dim], self.max_actions[dim])
            discretized_bins = np.digitize(clipped_action, self.bins[dim]) - 1
            discretized_bins = np.clip(discretized_bins, 0, self.n_bins - 1)
            mapper = np.vectorize(self.action_to_token_id.get)
            token_ids[:, dim] = mapper(discretized_bins)
        
        # 恢复原始形状
        if is_single:
            token_ids = token_ids.squeeze(0)
        
        return token_ids

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        将token IDs解码为动作值
        """
        action_token_ids = np.array(action_token_ids)
        if len(action_token_ids.shape) == 1: # 单个动作: [action_dims]
            action_token_ids = action_token_ids.reshape(1, -1)
            is_single = True
        else: is_single = False # 批量动作: [batch_size, action_dims]
        
        batch_size, action_dims = action_token_ids.shape
        actions = np.zeros_like(action_token_ids, dtype=float)
        
        for dim in range(action_dims):
            mapper = np.vectorize(self.token_id_to_action.get)
            discretized_actions = mapper(action_token_ids[:, dim])
            actions[:, dim] = self.bin_centers[dim][discretized_actions]
        
        # 恢复原始形状
        if is_single:
            actions = actions.squeeze(0)
        
        return actions

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        token_ids = self.encode(action)
        is_batch = len(token_ids.shape) > 1
        
        if is_batch:
            batch_strings = []
            for i in range(token_ids.shape[0]):
                sample_tokens = token_ids[i].flatten().tolist()
                batch_strings.append(self.tokenizer.decode(sample_tokens, skip_special_tokens=True))
            return batch_strings
        else:
            return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.n_bins
    
    @property
    def bos_token_id(self):
        """返回 BOS token ID"""
        return self.tokenizer.bos_token_id
    
    @property
    def eos_token_id(self):
        """返回 EOS token ID"""
        return self.tokenizer.eos_token_id
    
    @property
    def pad_token_id(self):
        """返回 PAD token ID"""
        return self.tokenizer.pad_token_id


class VLACofig(VLMConfig):
    model_type = "microvla"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            map_path: str = 'model/action_token_map.json',
            stats_path: str = 'dataset/stats.json',
            task_file_path: str = 'dataset/meta/tasks.jsonl',
            bins: int = 256,
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.map_path = map_path
        self.stats_path = stats_path
        self.bins = bins
        self.task_file_path = task_file_path
        super().__init__(**kwargs)


class MicroVLA(MiniMindVLM):
    config_class = VLACofig

    def __init__(self, params: VLACofig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        super().__init__(params, vision_model_path)
        if not params: params = VLACofig()
        self.params = params

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,    # [bs, seq_len]
                attention_mask: Optional[torch.Tensor] = None, 
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                pixel_values: Optional[torch.FloatTensor] = None, # [bs, num, c, im_h, im_w]
                **args):
        batch_size, seq_length_text = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids)) # [bs, seq_len, hidden_size]

        if len(pixel_values.shape) == 6:
            pixel_values = pixel_values.squeeze(2)
        bs, num, c, im_h, im_w = pixel_values.shape
        vision_features_list = []
        for i in range(num):
            img_features = MicroVLA.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
            vision_features_list.append(img_features)
        vision_tensors = torch.cat(vision_features_list, dim=1)  # [bs, num_img*196, hidden_size]

        # 3. 按顺序拼接: <bos> + vision_features + remaining_tokens
        hidden_states = torch.cat([
            hidden_states[:, 0:1, :],           # [bs, 1, hidden_size]
            vision_tensors,                     # [bs, num_img*196, hidden_size]
            hidden_states[:, 1:, :]             # [bs, seq_len-1, hidden_size]
        ], dim=1)                               # [bs, num_img*196+seq_len, hidden_size]
        seq_length = hidden_states.shape[1]
        
        # 在attention map合并之前先弄好labels 将pad_token的位置设为IGNORE_INDEX
        text_labels = input_ids[:, 1:].clone()
        text_attention = attention_mask[:, 1:]
        text_labels[text_attention == 0] = IGNORE_INDEX  # 将 pad 位置设为忽略

        # 创建图像部分的 attention_mask
        vision_attention_mask = torch.ones(
            (batch_size, vision_tensors.shape[1]), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        # 拼接新的 attention_mask
        attention_mask = torch.cat([
            attention_mask[:, 0:1],          # <bos> 的 mask
            vision_attention_mask,           # 图像 token 的 mask
            attention_mask[:, 1:]            # 原始文本 token 的 mask
        ], dim=1)

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states) # [bs, seq_len, vocab_size]

        # 构建完整的标签序列，包含text tokens和action tokens
        full_labels1 = torch.cat([
            torch.full((batch_size, 1), IGNORE_INDEX, 
                    dtype=input_ids.dtype, device=input_ids.device),        # <BOS> token (忽略)
            torch.full((batch_size, num * 196), IGNORE_INDEX, 
                    dtype=input_ids.dtype, device=input_ids.device),        # Vision tokens (忽略)
            text_labels,                                         # Text tokens and action tokens (参与loss)
        ], dim=1)

        # 创建action token mask
        action_only_labels = text_labels.clone()
        action_mask = torch.zeros_like(action_only_labels, dtype=torch.bool)
        # action_token_ids = set(action_tokenizer.action_to_token_id.values())
        # for token_id in action_token_ids:
        #     action_mask |= (action_only_labels == token_id)
        
        # # 将非action tokens设为IGNORE_INDEX
        # action_only_labels[~action_mask] = IGNORE_INDEX
        
        # full_labels2 = torch.cat([
        #     torch.full((batch_size, 1), IGNORE_INDEX, 
        #             dtype=input_ids.dtype, device=input_ids.device),        # <BOS> token (忽略)
        #     torch.full((batch_size, num * 196), IGNORE_INDEX, 
        #             dtype=input_ids.dtype, device=input_ids.device),        # Vision tokens (忽略)
        #     action_only_labels,                                  # 只有 Action tokens 参与loss，Text tokens被忽略
        # ], dim=1)

        # 实现时间错位
        shift_logits = logits[:, :-1, :].contiguous()    # 预测：位置 0 到 n-1
        shift_labels = full_labels1[:, 1:].contiguous()   # 目标：位置 1 到 n
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1)
        )
        self.OUT.__setitem__('loss', loss)
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT