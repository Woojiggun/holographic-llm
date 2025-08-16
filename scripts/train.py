"""
메인 학습 실행
"""
import sys
sys.path.append('.')

import torch
from configs.mvp_config import MVPConfig
from models.holographic_llm import HolographicLLM
from data.tokenizer import TokenizerManager
from data.dataset import create_dataloader, collate_fn
from training.trainer import Trainer

def main():
    print("="*60)
    print("Holographic LLM Training Start")
    print("="*60)
    
    # Config
    config = MVPConfig()
    print(f"[OK] Config loaded")
    
    # Tokenizer
    tokenizer = TokenizerManager(config)
    print(f"[OK] Tokenizer loaded (vocab: {config.vocab_size})")
    
    # Model
    model = HolographicLLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model created")
    print(f"  - Total: {total_params/1e6:.1f}M")
    print(f"  - Trainable: {trainable/1e6:.1f}M")
    
    # Data
    try:
        train_loader = create_dataloader('data/train.jsonl', tokenizer, config)
        val_loader = create_dataloader('data/val.jsonl', tokenizer, config)
        print(f"[OK] Data loaded successfully")
    except:
        print(f"[WARNING] No data files, using dummy data")
        # 더미 데이터로 대체
        from data.dataset import TextDataset
        train_dataset = TextDataset('dummy', config.max_seq_length)
        train_dataset.data = ["This is a training sentence."] * 1000
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, tokenizer.tokenizer, config.max_seq_length)
        )
        val_loader = None
    
    # GPU check
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"[WARNING] No GPU, CPU training (very slow)")
    
    # Trainer
    trainer = Trainer(model, config, train_loader, val_loader)
    
    print("\n" + "="*60)
    print("Training starting...")
    print("="*60)
    
    # OOM 안전장치
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\n[WARNING] OOM occurred! Settings need adjustment:")
        print("  - Reduce batch_size")
        print("  - Reduce max_seq_length")
        print("  - Increase gradient_accumulation_steps")
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted")
        print(f"Last checkpoint: {config.output_dir}")
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
    
    print("\n[COMPLETE] Done!")

if __name__ == "__main__":
    main()