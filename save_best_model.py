"""
최적 모델 저장
"""
import shutil
import os

# 체크포인트 300을 best_model.pt로 복사
source = 'outputs/checkpoint_300.pt'
dest = 'outputs/best_model.pt'

if os.path.exists(source):
    shutil.copy2(source, dest)
    print(f"✓ {source} → {dest} 복사 완료")
    print("체크포인트 300을 최적 모델로 저장했습니다.")
    print("\n사용법:")
    print("  python scripts/generate.py --checkpoint outputs/best_model.pt --prompt \"Your text\"")
else:
    print(f"✗ {source} 파일을 찾을 수 없습니다.")