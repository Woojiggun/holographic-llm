"""
홀로그래픽 어텐션 - 모든 버그 수정
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HolographicAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        # QKV 프로젝션
        self.qkv_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # 블렌딩 게이트 (안전한 초기값)
        self.blend_gate = nn.Parameter(torch.tensor(config.holographic_strength))
        
        # 정규화
        self.norm_attn = nn.LayerNorm(self.hidden_dim)
        self.norm_holo = nn.LayerNorm(self.hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 안전장치
        self.config = config
    
    def apply_rope(self, x, cos, sin):
        """RoPE 적용 - 올바른 shape"""
        # x: [batch, heads, seq, head_dim]
        # cos, sin: [1, 1, seq, head_dim//2]
        
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # 브로드캐스트 자동 처리
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        
        return torch.stack([rx1, rx2], dim=-1).flatten(-2)
    
    def create_causal_mask(self, seq_len, device):
        """Causal mask - bool 타입"""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    
    def standard_attention(self, q, k, v, mask=None):
        """표준 어텐션 - 안전한 마스킹"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 스코어 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 마스킹 (스코어 레벨에서만!)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 출력
        output = torch.matmul(attn_weights, v)
        return output
    
    def holographic_attention(self, q, k, v):
        """홀로그래픽 어텐션 - FP32 처리"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # FP32로 변환 (필수!)
        q_float = q.float()
        k_float = k.float()
        v_float = v.float()
        
        # 1D FFT (안전하게)
        try:
            q_freq = torch.fft.rfft(q_float, dim=-1, norm='ortho')
            k_freq = torch.fft.rfft(k_float, dim=-1, norm='ortho')
            v_freq = torch.fft.rfft(v_float, dim=-1, norm='ortho')
            
            # 간섭 패턴
            interference = torch.conj(q_freq) * k_freq
            modulated = interference * v_freq
            
            # 역변환
            output = torch.fft.irfft(modulated, n=head_dim, dim=-1, norm='ortho')
            
            # 원래 dtype으로 복원
            output = output.to(q.dtype)
            
        except Exception as e:
            print(f"FFT failed: {e}, using standard attention")
            output = self.standard_attention(q, k, v)
        
        return output
    
    def forward(self, x, attention_mask=None, cos=None, sin=None):
        batch_size, seq_len, _ = x.shape
        
        # QKV 계산
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # RoPE 적용
        if cos is not None and sin is not None:
            q = self.apply_rope(q, cos, sin)
            k = self.apply_rope(k, cos, sin)
        
        # Causal mask
        if attention_mask is None:
            attention_mask = self.create_causal_mask(seq_len, x.device)
        
        # 표준 어텐션
        attn_output = self.standard_attention(q, k, v, attention_mask)
        
        # 홀로그래픽 어텐션
        holo_output = self.holographic_attention(q, k, v)
        
        # Reshape
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        holo_output = holo_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        
        # 정규화
        attn_output = self.norm_attn(attn_output)
        holo_output = self.norm_holo(holo_output)
        
        # 안전한 블렌딩
        gate = torch.sigmoid(self.blend_gate)
        gate = torch.clamp(gate, self.config.gate_clamp_min, self.config.gate_clamp_max)
        output = (1 - gate) * attn_output + gate * holo_output
        
        # 출력 프로젝션
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output