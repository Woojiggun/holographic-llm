"""
메인 학습 스크립트
"""
import sys
import os
sys.path.append('.')

import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from configs.base_config import BaseConfig
from models.holographic_llm import HolographicLLM
from data.dataset import TextDataset
from training.trainer import Trainer

def collate_fn(batch, tokenizer, max_length):
    """배치 처리 함수"""
    texts = batch
    
    # 토큰화
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Labels 생성
    labels = encoded['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': labels
    }

def main():
    print("="*60)
    print("Holographic LLM V2 학습 시작")
    print("="*60)
    
    # Config
    config = BaseConfig()
    print(f"\n[Config]")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Save every: {config.save_steps} steps")
    
    # Tokenizer
    print(f"\n[Tokenizer 로드]")
    from transformers import LlamaTokenizer
    try:
        tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
        print(f"  LLaMA tokenizer loaded")
    except Exception as e:
        print(f"  LLaMA 로드 실패, GPT2 사용: {e}")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Config vocab_size 맞추기 (중요!)
    config.vocab_size = len(tokenizer)
    
    # Model
    print(f"\n[Model 생성]")
    model = HolographicLLM(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params/1e6:.1f}M")
    
    # Dataset
    print(f"\n[Dataset 로드]")
    train_dataset = TextDataset('data/train_real.jsonl', config.max_seq_length)
    val_dataset = TextDataset('data/val_real.jsonl', config.max_seq_length)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, config.max_seq_length),
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, config.max_seq_length),
        num_workers=0,
        pin_memory=True
    )
    
    # Trainer
    print(f"\n[Trainer 초기화]")
    trainer = Trainer(model, config, train_loader, val_loader)
    
    # 학습
    print(f"\n[학습 시작]")
    print("="*60)
    
    try:
        trainer.train()
        print("\n✓ 학습 완료!")
    except KeyboardInterrupt:
        print("\n⚠ 학습 중단됨")
        print(f"마지막 체크포인트: {config.output_dir}")
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()