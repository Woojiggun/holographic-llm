"""
중단된 학습 이어하기 스크립트
"""
import os
import sys
sys.path.append('.')

import torch
import glob
from pathlib import Path

def find_latest_checkpoint(output_dir):
    """가장 최근 체크포인트 찾기"""
    if not os.path.exists(output_dir):
        return None
    
    # 모든 체크포인트 파일 찾기
    pattern = os.path.join(output_dir, "checkpoint_*.pt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # 스텝 번호로 정렬
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    return checkpoints[-1]  # 가장 높은 스텝

def manual_save_checkpoint(model, optimizer, scheduler, scaler, step, loss, config, output_dir):
    """수동으로 체크포인트 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else {},
        'loss': loss,
        'config': config
    }
    
    # 현재 스텝 저장
    current_path = os.path.join(output_dir, f'checkpoint_{step}.pt')
    torch.save(checkpoint, current_path)
    print(f"[SAVED] Checkpoint: {current_path}")
    
    # 베스트 모델도 업데이트 (임시)
    best_path = os.path.join(output_dir, 'best_model_manual.pt')
    torch.save(checkpoint, best_path)
    print(f"[SAVED] Manual best: {best_path}")

def emergency_save_current_model():
    """현재 메모리에 있는 모델 저장 (학습 중단 시)"""
    print("="*60)
    print("Emergency Model Save")
    print("="*60)
    
    print("This function would save a model if training is currently running.")
    print("Since we can't access the training loop from here,")
    print("you need to manually add save code to your training script.")
    print("\nAdd this code where training was interrupted:")
    print("""
# Emergency save
emergency_checkpoint = {
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': current_loss,
    'config': config
}
torch.save(emergency_checkpoint, 'outputs_gpu/emergency_save.pt')
print("Emergency checkpoint saved!")
    """)

def main():
    print("="*60)
    print("Resume Training Helper")
    print("="*60)
    
    # 가능한 출력 디렉토리들 확인
    output_dirs = ['./outputs', './outputs_gpu']
    
    print("\n[Checkpoint Search]")
    for output_dir in output_dirs:
        print(f"\nChecking: {output_dir}")
        
        if not os.path.exists(output_dir):
            print(f"  Directory not found")
            continue
        
        # 체크포인트 파일들 찾기
        checkpoints = glob.glob(os.path.join(output_dir, "*.pt"))
        
        if checkpoints:
            print(f"  Found {len(checkpoints)} files:")
            for ckpt in sorted(checkpoints):
                basename = os.path.basename(ckpt)
                size_mb = os.path.getsize(ckpt) / (1024 * 1024)
                
                # 체크포인트에서 스텝 정보 추출 시도
                try:
                    checkpoint = torch.load(ckpt, map_location='cpu')
                    step = checkpoint.get('step', '?')
                    loss = checkpoint.get('loss', '?')
                    print(f"    {basename} ({size_mb:.1f}MB) - Step: {step}, Loss: {loss}")
                except:
                    print(f"    {basename} ({size_mb:.1f}MB) - Unable to read")
        else:
            print(f"  No checkpoint files found")
    
    print("\n" + "="*60)
    print("Resume Options")
    print("="*60)
    
    print("\n1. Create a new training script with frequent saves:")
    print("   python scripts/train_gpu_frequent_save.py")
    print("   - Saves every 50 steps")
    print("   - Won't lose progress easily")
    
    print("\n2. If you have W&B logs, check for the last logged step:")
    print("   Look in wandb/offline-run-*/files/output.log")
    
    print("\n3. For emergency save (if training is still running):")
    print("   Add save code to your training script")
    
    print("\n4. Resume from existing checkpoint:")
    
    # 최신 체크포인트 찾기
    latest_ckpt = find_latest_checkpoint('./outputs_gpu')
    if not latest_ckpt:
        latest_ckpt = find_latest_checkpoint('./outputs')
    
    if latest_ckpt:
        print(f"   Latest found: {latest_ckpt}")
        try:
            ckpt = torch.load(latest_ckpt, map_location='cpu')
            step = ckpt.get('step', 0)
            loss = ckpt.get('loss', 0)
            print(f"   Step: {step}, Loss: {loss:.4f}")
            
            print("\n   To resume from this checkpoint:")
            print(f"   python scripts/resume_from_checkpoint.py --checkpoint {latest_ckpt}")
        except Exception as e:
            print(f"   Error reading checkpoint: {e}")
    else:
        print("   No checkpoints found to resume from")

if __name__ == "__main__":
    main()