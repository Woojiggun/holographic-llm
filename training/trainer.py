"""
Trainer 클래스 - 간단하고 안정적인 버전
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, config, train_loader, val_loader=None):
        self.device = config.device
        self.model = model.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 시드 고정
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # Mixed Precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # 디렉토리
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch):
        """학습 스텝"""
        self.model.train()
        
        # Forward
        if self.config.use_amp and self.scaler:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs['loss']
        else:
            outputs = self.model(**batch)
            loss = outputs['loss']
        
        # Backward
        loss = loss / self.config.gradient_accumulation_steps
        
        if self.config.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def save_checkpoint(self, step=None, loss=None):
        """체크포인트 저장"""
        if step is None:
            step = self.global_step
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step,
            'loss': loss if loss is not None else 0.0,
            'config': {
                'model_name': self.config.model_name,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'vocab_size': self.config.vocab_size,
            }
        }
        
        path = os.path.join(self.config.output_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Saved checkpoint_{step}.pt")
        
        return path
    
    def train(self):
        """학습 메인 루프"""
        print(f"\n[Training Start]")
        print(f"  Device: {self.device}")
        print(f"  Steps: {self.config.max_steps}")
        print(f"  Save every: {self.config.save_steps} steps")
        
        accumulated_loss = 0
        accumulation_steps = 0
        
        progress = tqdm(total=self.config.max_steps, desc="Training")
        
        for epoch in range(100):
            for batch in self.train_loader:
                # 데이터 이동
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 학습
                loss = self.train_step(batch)
                accumulated_loss += loss
                accumulation_steps += 1
                
                # Gradient accumulation
                if accumulation_steps >= self.config.gradient_accumulation_steps:
                    # Optimizer step
                    if self.config.use_amp and self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1
                    
                    # 로깅
                    avg_loss = accumulated_loss / accumulation_steps
                    progress.update(1)
                    progress.set_postfix({'loss': f'{avg_loss:.4f}'})
                    
                    # 리셋
                    accumulated_loss = 0
                    accumulation_steps = 0
                    
                    # 체크포인트 저장
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(loss=avg_loss)
                    
                    # 최대 스텝
                    if self.global_step >= self.config.max_steps:
                        print(f"\n[Complete] Reached max steps: {self.config.max_steps}")
                        return
        
        print(f"\n[Complete] Training finished!")