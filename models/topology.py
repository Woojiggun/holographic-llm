"""
위상정보 - GPU 전용 + 안전장치
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientTopologyProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.sampling_rate = config.topology_sampling_rate
        self.k_neighbors = config.topology_k_neighbors
        
        # 경량 인코더
        self.topology_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 메트릭 임베딩
        self.metric_embeddings = nn.ModuleDict({
            'density': nn.Linear(1, 64),
            'connectivity': nn.Linear(1, 64),
            'clustering': nn.Linear(1, 64)
        })
        
        # 통합
        self.integration = nn.Linear(128 + 192, self.hidden_dim)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def compute_topology_metrics_gpu(self, embeddings):
        """GPU 전용 메트릭 계산"""
        batch_size, num_tokens, hidden_dim = embeddings.shape
        
        # 정규화
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        
        # 유사도 (작은 배치는 전체, 큰 배치는 샘플)
        if num_tokens <= 64:
            similarity = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2))
        else:
            # 샘플링
            sample_idx = torch.randperm(num_tokens)[:64]
            embeddings_sample = embeddings_norm[:, sample_idx, :]
            similarity = torch.bmm(embeddings_sample, embeddings_sample.transpose(1, 2))
        
        # 안전한 메트릭 계산
        with torch.no_grad():
            # 밀도
            density = similarity.mean(dim=(1, 2))
            
            # 연결성
            threshold = 0.5
            connectivity = (similarity > threshold).float().mean(dim=(1, 2))
            
            # 클러스터링
            clustering = (similarity ** 2).mean(dim=(1, 2))
        
        return {
            'density': density.unsqueeze(-1),
            'connectivity': connectivity.unsqueeze(-1),
            'clustering': clustering.unsqueeze(-1)
        }
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # 샘플링
        sample_size = min(64, max(16, int(seq_len * self.sampling_rate)))
        
        if seq_len > sample_size:
            indices = torch.randperm(seq_len, device=x.device)[:sample_size]
            sampled = x[:, indices, :]
        else:
            sampled = x
        
        # 특징 추출
        topology_features = self.topology_encoder(sampled.mean(dim=1))
        
        # 메트릭 계산
        metrics = self.compute_topology_metrics_gpu(sampled)
        
        # 임베딩
        metric_embeds = []
        for name, value in metrics.items():
            embed = self.metric_embeddings[name](value)
            metric_embeds.append(embed)
        
        metric_embeds = torch.cat(metric_embeds, dim=-1)
        
        # 통합
        combined = torch.cat([topology_features, metric_embeds], dim=-1)
        output = self.integration(combined)
        output = output.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 잔차
        return x + self.residual_weight * output