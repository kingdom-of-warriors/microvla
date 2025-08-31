import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from trainer.config import TrainingConfig, parse_args
from trainer.ddp_utils import setup_ddp, cleanup_ddp, reduce_metrics
from trainer.data_utils import create_dataloaders
from trainer.model_utils import create_vla_model
from trainer.metrics import compute_action_accuracy
from utils import load_stats

class VLATrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.rank, self.local_rank, self.world_size = setup_ddp()
        self.device = torch.device(f"cuda:{self.local_rank}")
        if self.rank == 0 and self.config.use_wandb:
            self._init_wandb()
        self.model, self.action_tokenizer, self.optimizer = None, None, None
        self.train_loader, self.val_loader, self.train_sampler = None, None, None

    def _init_wandb(self):
        wandb.init(
            project="microvla_debug",
            name=f"20250831_state",
            config=self.config.__dict__
        )

    def setup(self):
        """准备模型、数据和优化器"""
        # 创建模型和分词器
        action_stats = load_stats('dataset/meta/stats.json')['actions']
        model, self.action_tokenizer = create_vla_model(self.config.vla_config, 
                                                        action_stats, 
                                                        self.rank, 
                                                        tokenizer_path="model", 
                                                        ckpt_path="MiniMind2-V")
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=True)

        # 创建数据加载器和优化器
        self.train_loader, self.val_loader, self.train_sampler = create_dataloaders(
            self.config, self.action_tokenizer, self.rank, self.world_size
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        if self.rank == 0:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable params per GPU: {trainable_params / 1e6:.3f}M")

    def _run_epoch(self, epoch: int):
        self.train_sampler.set_epoch(epoch)
        self.model.train()
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", disable=(self.rank != 0))
        
        train_loss_sum, train_acc_sum, train_batches_sum = 0.0, 0.0, 0
        for batch in train_pbar:
            self.optimizer.zero_grad()
            
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, use_text_token=False)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            train_loss_sum += loss.item()
            acc = compute_action_accuracy(outputs, batch, self.action_tokenizer)
            if acc is not None:
                train_acc_sum += acc.item()
                train_batches_sum += 1
            
            if self.rank == 0:
                train_pbar.set_postfix({'loss': loss.item()})

        # 验证阶段
        self.model.eval()
        val_loss_sum, val_acc_sum, val_batches_sum = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", disable=(self.rank != 0)):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                val_loss_sum += outputs.loss.item()
                acc = compute_action_accuracy(outputs, batch, self.action_tokenizer)
                if acc is not None:
                    val_acc_sum += acc.item()
                    val_batches_sum += 1
        
        # 同步所有进程的指标
        train_metrics = torch.tensor([train_loss_sum, train_acc_sum, train_batches_sum, len(self.train_loader)], device=self.device)
        val_metrics = torch.tensor([val_loss_sum, val_acc_sum, val_batches_sum, len(self.val_loader)], device=self.device)
        train_metrics = reduce_metrics(train_metrics, self.world_size)
        val_metrics = reduce_metrics(val_metrics, self.world_size)

        if self.rank == 0:
            self._log_metrics(epoch, train_metrics.tolist(), val_metrics.tolist())

    def _log_metrics(self, epoch, train_stats, val_stats):
        # 计算平均指标
        avg_train_loss = train_stats[0] / train_stats[3]
        avg_train_acc = train_stats[1] / train_stats[2] if train_stats[2] > 0 else 0
        avg_val_loss = val_stats[0] / val_stats[3]
        avg_val_acc = val_stats[1] / val_stats[2] if val_stats[2] > 0 else 0

        # 打印到控制台
        print(f"\n--- Epoch {epoch+1} Metrics ---")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}, Avg Train Acc: {avg_train_acc:.4f}")
        print(f"  Avg Val Loss:   {avg_val_loss:.4f}, Avg Val Acc:   {avg_val_acc:.4f}")
        print("-" * 50)

        # 记录到wandb
        if self.config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": avg_train_loss, "train/epoch_action_acc": avg_train_acc,
                "val/epoch_loss": avg_val_loss, "val/epoch_action_acc": avg_val_acc
            })

    def _save_checkpoint(self, epoch):
        if self.rank == 0:
            os.makedirs("ckpt", exist_ok=True)
            checkpoint_path = f"ckpt/state/en_{epoch+1}.pth"
            torch.save(self.model.module.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    def train(self):
        self.setup()
        for epoch in range(self.config.epochs):
            self._run_epoch(epoch)
            self._save_checkpoint(epoch)
        
        if self.rank == 0 and self.config.use_wandb:
            wandb.finish()
        cleanup_ddp()

def main():
    config = parse_args()
    trainer = VLATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()