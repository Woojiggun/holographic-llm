"""
동적 범주 - EMA 통계 + 안전장치
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicCategoryMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_size = config.intermediate_size
        self.num_games = config.num_language_games
        
        # 라우터
        self.router = nn.Linear(self.hidden_dim, self.num_games)
        
        # 전문가들
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.intermediate_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.intermediate_size, self.hidden_dim)
            )
            for _ in range(self.num_games)
        ])
        
        # 언어게임 임베딩
        self.game_embeddings = nn.Parameter(
            torch.randn(self.num_games, self.hidden_dim) * 0.02
        )
        
        # EMA 통계
        self.register_buffer('expert_counts', torch.ones(self.num_games))
        self.register_buffer('_update_counter', torch.tensor(0))
        self.ema_decay = 0.9
    
    def update_statistics(self, indices):
        """EMA 통계 업데이트"""
        if self.training:
            # 카운트
            for i in range(self.num_games):
                count = (indices == i).float().sum()
                self.expert_counts[i] = self.ema_decay * self.expert_counts[i] + (1 - self.ema_decay) * count
            
            # 주기적 정규화
            self._update_counter += 1
            if self._update_counter % 100 == 0:
                self.expert_counts = self.expert_counts / (self.expert_counts.sum() + 1e-6)
                self.expert_counts = self.expert_counts * self.num_games
    
    def forward(self, x, topology_info=None):
        batch_size, seq_len, hidden_dim = x.shape
        
        # 라우팅
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-1
        expert_indices = router_probs.argmax(dim=-1)
        
        # 통계
        self.update_statistics(expert_indices.flatten())
        
        # 배치 처리
        output = torch.zeros_like(x)
        
        for idx in range(self.num_games):
            mask = (expert_indices == idx)
            if mask.any():
                selected = x[mask]
                selected = selected + 0.1 * self.game_embeddings[idx]
                expert_output = self.experts[idx](selected)
                # dtype 맞추기 (mixed precision 대응)
                output[mask] = expert_output.to(output.dtype)
        
        # 위상 정보
        if topology_info is not None:
            output = output + 0.1 * topology_info
        
        # Load balancing loss
        aux_loss = None
        if self.training:
            target = 1.0 / self.num_games
            actual = self.expert_counts / (self.expert_counts.sum() + 1e-6)
            aux_loss = 0.01 * F.mse_loss(actual, torch.full_like(actual, target))
        
        return output, aux_loss