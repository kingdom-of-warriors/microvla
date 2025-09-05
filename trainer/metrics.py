import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_vla import ActionTokenizer

# DEBUG
def compute_action_accuracy(outputs, batch, action_tokenizer: ActionTokenizer, num_vision_patches=196):
    """计算动作预测的准确率"""
    # # pred logits中，跳过vision token和instruction token部分
    # action_logits = outputs['logits'][:, 2 * num_vision_patches + 1:, ]
    # action_preds = action_logits.argmax(dim=-1)
    # # GT labels中，跳过BOS token
    # action_gt = batch['input_ids'][:, 1:].to(action_preds.device)
    # # 创建一个mask，只关注action token的预测准确率
    # action_token_ids = set(action_tokenizer.action_to_token_id.values())
    # mask = torch.zeros_like(action_gt, dtype=torch.bool)
    # for token_id in action_token_ids:
    #     mask |= (action_gt == token_id)
    
    # if mask.sum() == 0: 
    #     return None
    
    # correct_preds = (action_preds == action_gt) & mask
    # return correct_preds.sum().float() / mask.sum().float()
    return torch.tensor(0.0)