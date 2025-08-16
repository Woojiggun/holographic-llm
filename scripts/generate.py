"""
텍스트 생성 스크립트
"""
import sys
sys.path.append('.')

import torch
import argparse
from pathlib import Path

def load_model_from_checkpoint(checkpoint_path):
    """체크포인트에서 모델 로드"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # config 생성
    from configs.base_config import BaseConfig
    config = BaseConfig()
    
    # 체크포인트에서 config 정보가 있으면 사용
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 모델 생성
    from models.holographic_llm import HolographicLLM
    model = HolographicLLM(config)
    
    # 가중치 로드 (키 이름 체크)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("No model weights found in checkpoint")
    
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Training step: {checkpoint.get('step', 'unknown')}")
    
    return model, config

def generate_text(model, tokenizer, prompt, 
                  max_length=100, 
                  temperature=0.8, 
                  top_k=50, 
                  top_p=0.95,
                  device='cpu'):
    """텍스트 생성"""
    model.to(device)
    model.eval()
    
    # 프롬프트 토큰화
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    
    try:
        print(f"\nPrompt: {prompt}")
    except UnicodeEncodeError:
        print(f"\nPrompt: {prompt.encode('utf-8', errors='replace').decode('utf-8')}")
    print(f"Generating (max {max_length} tokens)...")
    print("-" * 50)
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_ids=generated)
            logits = outputs['logits']
            
            # 마지막 토큰의 logits
            next_token_logits = logits[0, -1, :]
            
            # Temperature 적용
            next_token_logits = next_token_logits / temperature
            
            # Top-k 필터링
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Top-p (nucleus) 필터링
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Softmax로 확률 계산
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # 샘플링
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 생성된 토큰 추가
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # EOS 토큰 체크
            if hasattr(tokenizer, 'eos_token_id'):
                if next_token.item() == tokenizer.eos_token_id:
                    break
            elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'eos_token_id'):
                if next_token.item() == tokenizer.tokenizer.eos_token_id:
                    break
    
    # 디코딩
    if hasattr(tokenizer, 'decode'):
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    else:
        generated_text = tokenizer.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    return generated_text

def interactive_generation(model, tokenizer, config, device='cpu'):
    """대화형 텍스트 생성"""
    print("\n" + "="*60)
    print("Interactive Text Generation")
    print("="*60)
    print("Commands:")
    print("  /quit - Exit")
    print("  /temp <value> - Set temperature (0.1-2.0)")
    print("  /length <value> - Set max length (10-500)")
    print("  /reset - Reset conversation")
    print("="*60)
    
    temperature = 0.8
    max_length = 100
    
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
            
            if not prompt:
                continue
            
            # 명령어 처리
            if prompt == '/quit':
                break
            elif prompt.startswith('/temp'):
                try:
                    temperature = float(prompt.split()[1])
                    temperature = max(0.1, min(2.0, temperature))
                    print(f"Temperature set to {temperature}")
                except:
                    print("Usage: /temp <0.1-2.0>")
                continue
            elif prompt.startswith('/length'):
                try:
                    max_length = int(prompt.split()[1])
                    max_length = max(10, min(500, max_length))
                    print(f"Max length set to {max_length}")
                except:
                    print("Usage: /length <10-500>")
                continue
            elif prompt == '/reset':
                print("Conversation reset")
                continue
            
            # 텍스트 생성
            generated = generate_text(
                model, tokenizer, prompt,
                max_length=max_length,
                temperature=temperature,
                device=device
            )
            
            print(f"\nGenerated:\n{generated}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

def main():
    parser = argparse.ArgumentParser(description='Generate text with Holographic LLM')
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.1-2.0)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Top-p (nucleus) sampling')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # 체크포인트 확인
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # 모델 로드
    model, config = load_model_from_checkpoint(args.checkpoint)
    
    # 토크나이저 로드
    from transformers import LlamaTokenizer, GPT2TokenizerFast
    try:
        tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
        print(f"Using LLaMA tokenizer")
    except:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        print(f"Fallback to GPT2 tokenizer")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 생성 모드
    if args.interactive:
        interactive_generation(model, tokenizer, config, device)
    else:
        if args.prompt is None:
            # 기본 프롬프트들
            prompts = [
                "The holographic model",
                "In the future,",
                "Once upon a time",
                "The key to understanding",
            ]
            print(f"\n{'='*60}")
            print("Example Generations")
            print(f"{'='*60}")
            
            for prompt in prompts:
                generated = generate_text(
                    model, tokenizer, prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device
                )
                print(f"\n[Prompt: {prompt}]")
                print(f"{generated}")
                print("-" * 40)
        else:
            # 사용자 프롬프트
            generated = generate_text(
                model, tokenizer, args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device
            )
            try:
                print(f"\nGenerated text:\n{generated}")
            except UnicodeEncodeError:
                print(f"\nGenerated text:\n{generated.encode('utf-8', errors='replace').decode('utf-8')}")

if __name__ == "__main__":
    main()