"""
자주 저장하는 GPU 학습 스크립트
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # RTX 4070

import sys
sys.path.append('.')

import torch
from configs.gpu_frequent_save import GPUFrequentSaveConfig
from models.holographic_llm import HolographicLLM
from data.tokenizer import TokenizerManager
from data.dataset import create_dataloader, collate_fn
from training.trainer import Trainer

def main():
    print("="*60)
    print("GPU Training with Frequent Checkpoints (RTX 4070)")
    print("="*60)
    
    # GPU 확인
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] GPU: {gpu_name} ({memory:.1f} GB)")
    else:
        print("[ERROR] CUDA not available!")
        return
    
    # Config
    config = GPUFrequentSaveConfig()
    print(f"\n[Config]")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Save every: {config.save_steps} steps")
    print(f"  Eval every: {config.eval_steps} steps")
    print(f"  Log every: {config.logging_steps} steps")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps}")
    
    # 디렉토리 확인
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"  Directory created: {config.output_dir}")
    
    # Model & Data
    tokenizer = TokenizerManager(config)
    model = HolographicLLM(config)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] {params/1e6:.1f}M parameters")
    
    train_loader = create_dataloader('data/train.jsonl', tokenizer, config)
    val_loader = create_dataloader('data/val.jsonl', tokenizer, config)
    
    # Training
    print(f"\n[Starting Training]")
    print(f"Expected checkpoints: step 50, 100, 150, 200, 250, 300...")
    
    trainer = Trainer(model, config, train_loader, val_loader)
    
    try:
        trainer.train()
        print("\n[SUCCESS] Training completed!")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Training stopped")
        print(f"Check {config.output_dir} for saved checkpoints")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # 저장된 체크포인트 확인
    if os.path.exists(config.output_dir):
        checkpoints = [f for f in os.listdir(config.output_dir) if f.endswith('.pt')]
        if checkpoints:
            print(f"\n[Saved Checkpoints]")
            for ckpt in sorted(checkpoints):
                ckpt_path = os.path.join(config.output_dir, ckpt)
                size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                print(f"  {ckpt} ({size_mb:.1f} MB)")
        else:
            print(f"\n[WARNING] No checkpoints found in {config.output_dir}")
    
    # Memory stats
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak memory: {peak_memory:.2f} GB / 12.0 GB")

if __name__ == "__main__":
    main()