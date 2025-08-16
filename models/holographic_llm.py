"""
메인 모델 - 모든 안전장치 포함
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.attention import HolographicAttention
from models.topology import EfficientTopologyProcessor
from models.moe_ffn import DynamicCategoryMoE
from models.utils import RMSNorm, RotaryEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.norm3 = RMSNorm(config.hidden_dim)
        
        self.attention = HolographicAttention(config)
        self.topology = EfficientTopologyProcessor(config)
        self.ffn = DynamicCategoryMoE(config)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, attention_mask, cos, sin):
        # Attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask, cos, sin)
        x = residual + self.dropout(x)
        
        # Topology
        residual = x
        x = self.norm2(x)
        x = self.topology(x)
        topology_info = x - residual
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm3(x)
        x, aux_loss = self.ffn(x, topology_info)
        x = residual + self.dropout(x)
        
        return x, aux_loss

class HolographicLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 임베딩
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_seq_length)
        
        # 레이어
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # 출력
        self.final_norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # 가중치 공유
        self.lm_head.weight = self.embedding.weight
        
        # 초기화
        self.apply(self._init_weights)
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 임베딩
        hidden_states = self.embedding(input_ids)
        
        # RoPE
        cos, sin = self.rotary_emb(seq_len)
        cos = cos.to(device)
        sin = sin.to(device)
        
        # attention_mask는 padding mask로만 사용 (causal mask는 attention 내부에서 생성)
        # None으로 전달하면 attention 레이어가 causal mask를 자동 생성
        
        # 레이어 통과
        aux_losses = []
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Checkpoint (텐서만 전달)
                hidden_states, aux_loss = checkpoint(
                    layer,
                    hidden_states,
                    None,  # attention_mask는 None으로 (causal은 내부에서)
                    cos,
                    sin
                )
            else:
                hidden_states, aux_loss = layer(hidden_states, None, cos, sin)  # None으로 전달
            
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        
        # 최종
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # 손실
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100  # 패딩 무시
            )
            
            # Aux losses
            if aux_losses:
                aux_loss = sum(aux_losses) / len(aux_losses)
                loss = loss + aux_loss
        
        return {'loss': loss, 'logits': logits}