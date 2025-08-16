"""
현재 학습 중인 모델 즉시 저장
"""
import torch
import os

def save_current_checkpoint():
    # 현재 학습 중인 프로세스에서 사용할 코드
    # trainer.py에 추가하거나 별도로 실행
    
    print("Manual checkpoint save utility")
    print("Add this to your training loop or run separately")
    
    code = '''
# trainer.py에 추가할 코드
# 또는 학습 중 Ctrl+C 후 실행

# 체크포인트 저장 함수
def save_checkpoint_now(trainer, name="manual_checkpoint"):
    checkpoint = {
        'step': trainer.global_step,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'loss': trainer.current_loss if hasattr(trainer, 'current_loss') else 0,
        'config': trainer.config
    }
    
    path = os.path.join(trainer.config.output_dir, f'{name}_{trainer.global_step}.pt')
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    
# 사용법:
# save_checkpoint_now(trainer, "gpu_model")
'''
    
    print(code)
    
    # 또는 별도 스크립트로
    print("\n" + "="*60)
    print("Alternative: Interrupt training and save")
    print("="*60)
    print("1. Press Ctrl+C to pause training")
    print("2. Run this in the same Python session:")
    print("   torch.save(model.state_dict(), 'outputs/gpu_model_current.pt')")

if __name__ == "__main__":
    save_current_checkpoint()
