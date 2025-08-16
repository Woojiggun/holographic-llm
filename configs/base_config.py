"""
기본 설정 - V1과 동일한 구조
"""
import torch

class BaseConfig:
    # 모델 아키텍처
    model_name = "HolographicLLM-V2"
    vocab_size = 32000  # LLaMA 기본값 (자동 조정됨)
    hidden_dim = 768
    num_layers = 12
    num_heads = 12
    head_dim = 64
    intermediate_size = 3072
    max_seq_length = 512
    
    # 학습 설정
    batch_size = 2
    gradient_accumulation_steps = 16
    learning_rate = 5e-5
    warmup_steps = 500
    max_steps = 10000
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # 홀로그래픽 설정
    holographic_strength = 0.1  # 안전한 시작값
    topology_sampling_rate = 0.05
    topology_k_neighbors = 4
    num_language_games = 3
    
    # 정규화
    dropout = 0.1
    attention_dropout = 0.1
    label_smoothing = 0.1
    
    # 안전장치
    seed = 42
    nan_detection = True
    gate_clamp_min = 0.05
    gate_clamp_max = 0.9
    
    # 경로
    output_dir = "./outputs"
    save_steps = 50
    eval_steps = 100
    logging_steps = 10
    
    # 시스템
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    gradient_checkpointing = False