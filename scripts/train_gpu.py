"""
GPU 학습용 스크립트 (RTX 4070 우선 사용)
"""
import sys
sys.path.append('.')

import torch
import os

# RTX 4070 우선 사용 설정 - 환경변수 설정 전에 torch import 필요
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 4070이 GPU 1번

from configs.mvp_config import MVPConfig
from models.holographic_llm import HolographicLLM
from data.tokenizer import TokenizerManager
from data.dataset import create_dataloader, collate_fn
from training.trainer import Trainer

def main():
    print("="*60)
    print("Holographic LLM Training (GPU Mode - RTX 4070)")
    print("="*60)
    
    # GPU 확인
    if torch.cuda.is_available():
        print(f"[OK] CUDA available")
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[ERROR] CUDA not available!")
        print("Please run: scripts/install_cuda_pytorch.bat")
        return
    
    # Config (GPU 최적화)
    config = MVPConfig()
    print(f"\n[OK] Config loaded")
    print(f"     Model: {config.model_name}")
    print(f"     Hidden dim: {config.hidden_dim}")
    print(f"     Layers: {config.num_layers}")
    print(f"     Batch size: {config.batch_size}")
    
    # Tokenizer
    tokenizer = TokenizerManager(config)
    print(f"[OK] Tokenizer loaded (vocab: {config.vocab_size})")
    
    # Model
    model = HolographicLLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model created")
    print(f"     Total: {total_params/1e6:.1f}M")
    print(f"     Trainable: {trainable/1e6:.1f}M")
    
    # 메모리 사용량 예측
    param_memory = total_params * 4 / 1024**3  # FP32
    optimizer_memory = param_memory * 2  # Adam states
    activation_memory = config.batch_size * config.max_seq_length * config.hidden_dim * config.num_layers * 4 / 1024**3
    total_memory = param_memory + optimizer_memory + activation_memory
    print(f"\n[Memory Estimate]")
    print(f"     Model: {param_memory:.2f} GB")
    print(f"     Optimizer: {optimizer_memory:.2f} GB")
    print(f"     Activations: {activation_memory:.2f} GB")
    print(f"     Total: {total_memory:.2f} GB")
    
    if total_memory > 11:
        print(f"[WARNING] Estimated memory ({total_memory:.2f} GB) exceeds RTX 4070 VRAM (12 GB)")
        print(f"          Consider reducing batch_size or using gradient_checkpointing")
    
    # Data
    try:
        train_loader = create_dataloader('data/train.jsonl', tokenizer, config)
        val_loader = create_dataloader('data/val.jsonl', tokenizer, config)
        print(f"\n[OK] Data loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
    
    # Trainer
    trainer = Trainer(model, config, train_loader, val_loader)
    
    print("\n" + "="*60)
    print("Training starting on RTX 4070...")
    print(f"Steps: {config.max_steps}")
    print(f"Mixed Precision: {config.mixed_precision}")
    print(f"Gradient Checkpointing: {config.gradient_checkpointing}")
    print("="*60)
    
    # Training
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] GPU OOM! Reducing batch size or sequence length needed")
        print("Current settings:")
        print(f"  batch_size: {config.batch_size}")
        print(f"  max_seq_length: {config.max_seq_length}")
        print(f"  gradient_accumulation: {config.gradient_accumulation_steps}")
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted")
        print(f"Last checkpoint: {config.output_dir}")
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[COMPLETE] Done!")

if __name__ == "__main__":
    # RTX 4070 강제 선택
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # GPU 1번(RTX 4070) 선택
        torch.cuda.set_device(1)
        print(f"Selected GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    main()