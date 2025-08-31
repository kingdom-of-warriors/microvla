import os
import numpy as np
import torch
import warnings
from .model_vlm import *
from typing import Optional, Tuple, List
from torch import nn

warnings.filterwarnings('ignore')

import json
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Union
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
        if len(action_token_ids.shape) == 1:                      # 单个动作: [action_dims]
            action_token_ids = action_token_ids.reshape(1, -1)
            is_single = True
        else: 
            is_single = False                                   # 批量动作: [batch_size, action_dims]
        
        batch_size, action_dims = action_token_ids.shape
        actions = np.zeros_like(action_token_ids, dtype=float)
        
        for dim in range(action_dims):
            mapper = np.vectorize(self.token_id_to_action.get)
            discretized_actions = mapper(action_token_ids[:, dim])
            discretized_actions = np.clip(discretized_actions, 0, self.bin_centers.shape[1] - 1) # 防止溢出
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
    def bos_token_id(self): return self.tokenizer.bos_token_id
    
    @property
    def eos_token_id(self): return self.tokenizer.eos_token_id
    
    @property
    def pad_token_id(self): return self.tokenizer.pad_token_id


class VLACofig(VLMConfig):
    model_type = "microvla"

    def __init__(
            self,
            map_path: str = 'model/action_token_map_256.json',
            stats_path: str = 'dataset/stats.json',
            task_file_path: str = 'dataset/meta/tasks.jsonl',
            bins: int = 256,
            state_dim: int = 8,
            use_state: bool = True,  # 是否使用本体感知
            **kwargs,
    ):
        self.map_path = map_path
        self.stats_path = stats_path
        self.bins = bins
        self.task_file_path = task_file_path
        self.state_dim = state_dim
        self.use_state = use_state
        super().__init__(**kwargs)


class MicroVLA(MiniMindVLM):
    config_class = VLACofig

    def __init__(self, params: VLACofig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16", action_tokenizer: ActionTokenizer = None):
        super().__init__(params, vision_model_path)
        if not params: params = VLACofig()
        self.params = params
        self.action_tokenizer = action_tokenizer
        if self.params.use_state:
            self.state_projector = nn.Sequential(
                nn.Linear(self.params.state_dim, self.params.hidden_size),
                nn.SiLU(),
                nn.Dropout(self.params.dropout),
                nn.Linear(self.params.hidden_size, self.params.hidden_size),
                RMSNorm(self.params.hidden_size, eps=self.params.rms_norm_eps)
            )
        else:
            self.state_projector = None

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,    # [bs, seq_len] 
                attention_mask: Optional[torch.Tensor] = None,  
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
                use_cache: bool = False, 
                pixel_values: Optional[torch.FloatTensor] = None, # [bs, num, c, im_h, im_w] 
                state_values: Optional[torch.FloatTensor] = None, # [bs, state_dim]
                use_text_token: bool = False, 
                **args) -> CausalLMOutputWithPast:

        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        loss = None
        # KV 缓存核心：start_pos 判断是预填充还是解码步骤。
        if start_pos == 0:
            # 1. 隐藏层初始化 
            hidden_states = self.model.dropout(self.model.embed_tokens(input_ids)) # [bs, seq_len, hidden_size]      
            # 2. 获取图像嵌入
            if len(pixel_values.shape) == 6: pixel_values = pixel_values.squeeze(2) 
            bs, num, c, im_h, im_w = pixel_values.shape 
            vision_features_list = [] 
            for i in range(num): 
                img_features = MicroVLA.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder) 
                if bs == 1: img_features = img_features.unsqueeze(0) # 补上bs维度
                vision_features_list.append(img_features) 
            vision_tensors = torch.cat(vision_features_list, dim=1)  # [bs, num_img*196, hidden_size] 

            # 3. 处理本体感知信息（新增）
            state_tensors = None
            if self.params.use_state and state_values is not None:
                state_features = self.state_projector(state_values)  # [bs, state_proj_dim]
                state_tensors = state_features.unsqueeze(1)  # [bs, 1, state_proj_dim]

            # 4. 按顺序拼接
            if state_tensors is not None: # 使用state
                hidden_states = torch.cat([ 
                    hidden_states[:, 0:1, :],        # <bos>
                    vision_tensors,                  # vision tokens
                    state_tensors,                   # state token (新增)
                    hidden_states[:, 1:, :]          # remaining tokens
                ], dim=1)
                
                # 5. 创建对应的 attention_mask
                vision_attention_mask = torch.ones( 
                    (bs, vision_tensors.shape[1]),  
                    dtype=attention_mask.dtype,  
                    device=attention_mask.device 
                )
                state_attention_mask = torch.ones(
                    (bs, 1),  # state 只有1个token
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([ 
                    attention_mask[:, 0:1],          # <bos> mask
                    vision_attention_mask,           # vision mask
                    state_attention_mask,            # state mask (新增)
                    attention_mask[:, 1:]            # remaining mask
                ], dim=1)
            else: # 不使用 state
                hidden_states = torch.cat([ 
                    hidden_states[:, 0:1, :],
                    vision_tensors,
                    hidden_states[:, 1:, :]
                ], dim=1)
                
                vision_attention_mask = torch.ones( 
                    (bs, vision_tensors.shape[1]),  
                    dtype=attention_mask.dtype,  
                    device=attention_mask.device 
                ) 
                attention_mask = torch.cat([ 
                    attention_mask[:, 0:1],
                    vision_attention_mask,
                    attention_mask[:, 1:]
                ], dim=1)
            
        # 利用已有的 KV 缓存，只需处理新的 token(s)，`attention_mask` 应该由外部的生成循环更新并传入
        else:
            hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))
            
        seq_length = hidden_states.shape[1]
        position_embeddings = ( 
            self.model.freqs_cos[start_pos : start_pos + seq_length], 
            self.model.freqs_sin[start_pos : start_pos + seq_length] 
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

        # Loss计算
        if start_pos == 0:
            bs, _ = input_ids.shape
            
            # label类型1：创建只包含 text + action 的标签序列
            vision_tokens_len = vision_tensors.shape[1]
            state_tokens_len = 1 if (self.params.use_state and state_values is not None) else 0
            prefix_len = 1 + vision_tokens_len + state_tokens_len  # <BOS> + vision + state
            
            text_labels = input_ids[:, 1:].clone()
            text_attention = attention_mask[:, prefix_len:]
            text_labels[text_attention == 0] = IGNORE_INDEX
            full_labels1 = torch.cat([ 
                torch.full((bs, 1 + vision_tensors.shape[1]), 
                           IGNORE_INDEX, 
                           dtype=input_ids.dtype, 
                           device=input_ids.device),
                text_labels,
            ], dim=1) 

            # label类型2：创建只包含 action 的标签序列
            action_only_labels = text_labels.clone() 
            action_mask = torch.zeros_like(action_only_labels, dtype=torch.bool) 
            action_token_ids = set(self.action_tokenizer.action_to_token_id.values()) 
            for token_id in action_token_ids: action_mask |= (action_only_labels == token_id) 
            action_only_labels[~action_mask] = IGNORE_INDEX 
            full_labels2 = torch.cat([ 
                torch.full((bs, 1 + vision_tensors.shape[1]), 
                           IGNORE_INDEX, 
                           dtype=input_ids.dtype, 
                           device=input_ids.device),
                action_only_labels,
            ], dim=1) 

            full_labels = full_labels1 if use_text_token else full_labels2 # 根据 use_text_token 选择用哪个标签计算Loss

            shift_logits = logits[:, :-1, :].contiguous()   # 预测：位置 0 到 n-1 
            shift_labels = full_labels[:, 1:].contiguous()  # 目标：位置 1 到 n
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) 
            loss = loss_fct( 
                shift_logits.view(-1, self.config.vocab_size), 
                shift_labels.view(-1) 
            )
        
        self.OUT['loss'] = loss
        self.OUT['hidden_states'] = hidden_states 
        self.OUT['logits'] = logits 
        # 只有当 use_cache 为 True 时才返回更新后的 KV 缓存
        self.OUT['past_key_values'] = presents if use_cache else None
        return self.OUT

    @torch.no_grad()
    def predict_action_kv_cache(self, pixel_values: torch.FloatTensor, state_values: Optional[torch.FloatTensor], task_description: str):
        """
        使用 VLA 模型【自回归地】预测一个完整的动作序列。
        """
        self.eval()
        device = self.device 
        batch_size = pixel_values.shape[0]
        action_dims = self.action_tokenizer.action_dims

        # 1. 准备初始的文本输入: <bos> + instruction
        instruction_ids = self.action_tokenizer.tokenizer.encode(
            task_description, 
            add_special_tokens=False, 
            return_tensors='pt'
        ).to(device)
        input_ids = instruction_ids.repeat(batch_size, 1)

        bos_id = self.action_tokenizer.bos_token_id
        bos_tensor = torch.full((batch_size, 1), bos_id, device=device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)

        past_key_values = None
        generated_action_ids = []
        
        # 预先获取有效的动作token ID列表
        action_token_ids_list = list(self.action_tokenizer.action_to_token_id.values())
        action_token_ids_tensor = torch.tensor(action_token_ids_list, device=device)

        # 2. 自回归生成N个动作token
        for _ in range(action_dims):
            attention_mask = torch.ones_like(input_ids)
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                pixel_values=pixel_values,
                state_values=state_values
            )

            next_token_logits = outputs.logits[:, -1, :]
            action_logits = torch.index_select(next_token_logits, dim=1, index=action_token_ids_tensor) # 约束解码：只在有效动作空间内选择
            best_action_index = torch.argmax(action_logits, dim=-1)
            predicted_token_id = action_token_ids_tensor[best_action_index]
            generated_action_ids.append(predicted_token_id.unsqueeze(1))
            input_ids = predicted_token_id.unsqueeze(1)
            past_key_values = outputs.past_key_values
            pixel_values = None # 在第一次迭代后，不再需要传入图像，其信息已在past_key_values中
            state_values = None 
            
        # 3. 将所有生成的动作token ID拼接起来
        action_token_ids_tensor = torch.cat(generated_action_ids, dim=1)
        actions_np = self.action_tokenizer.decode_token_ids_to_actions(
            action_token_ids_tensor.cpu().numpy()
        )
        
        return actions_np
    
    @torch.no_grad()
    def predict_action(self, pixel_values: torch.FloatTensor, task_description: str):
        """
        使用 VLA 模型【自回归地】预测一个完整的动作序列。
        【注意】此版本不使用 KV 缓存，效率较低，主要用于调试或对比。
        """
        self.eval()
        device = self.device
        batch_size = pixel_values.shape[0]
        action_dims = self.action_tokenizer.action_dims

        # 1. 准备初始的文本输入: <bos> + instruction
        instruction_ids = self.action_tokenizer.tokenizer.encode(
            task_description, add_special_tokens=False, return_tensors='pt'
        ).to(device)
        
        bos_id = self.action_tokenizer.bos_token_id
        bos_tensor = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        generated_ids = torch.cat([bos_tensor, instruction_ids.repeat(batch_size, 1)], dim=1)

        # 2. 预先获取有效的动作 token ID 列表，用于约束解码
        action_token_ids_list = list(self.action_tokenizer.action_to_token_id.values())
        action_token_ids_tensor = torch.tensor(action_token_ids_list, device=device)

        # 3. 自回归生成 N 个动作 token
        for _ in range(action_dims):
            attention_mask = torch.ones_like(generated_ids)
            outputs = self.forward(
                pixel_values=pixel_values,
                input_ids=generated_ids,
                attention_mask=attention_mask,
                use_cache=False,
                past_key_values=None
            )

            # c. 获取序列最后一个 token 的 logits，用于预测下一个 token
            next_token_logits = outputs['logits'][:, -1, :]
            action_logits = torch.index_select(next_token_logits, dim=1, index=action_token_ids_tensor)
            best_action_index = torch.argmax(action_logits, dim=-1)
            predicted_token_id = action_token_ids_tensor[best_action_index]
            
            # e. 将新生成的 token 拼接到序列末尾，为下一次迭代做准备
            generated_ids = torch.cat([
                generated_ids, 
                predicted_token_id.unsqueeze(1)
            ], dim=1)

        # 4. 提取出动作部分的 token ID
        action_token_ids_tensor = generated_ids[:, -action_dims:]
        # 5. 将 token ID 解码为实际的动作值
        actions_np = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids_tensor.cpu().numpy())
        
        return actions_np
