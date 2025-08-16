"""
유틸리티 - RoPE 올바른 shape
"""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True) + self.eps
        x = x * torch.rsqrt(norm)
        return self.weight * x.type_as(self.weight)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 미리 계산
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, seq_len):
        """올바른 shape 반환"""
        # [seq_len, dim//2]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # 브로드캐스트용 shape [1, 1, seq_len, dim//2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        return cos, sin