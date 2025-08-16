# 홀로그래픽 LLM: FFT 기반 어텐션 메커니즘 실험

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> 물리학의 홀로그래피 원리를 트랜스포머 어텐션 메커니즘에 적용한 실험적 언어 모델

## 🌟 핵심 혁신

이 프로젝트는 **홀로그래픽 어텐션**이라는 새로운 접근법을 소개합니다:

- **FFT 기반 간섭 패턴**: 주파수 도메인에서 어텐션 처리
- **위상 정보 인코딩**: 복소수 위상을 활용한 풍부한 표현
- **동적 카테고리 MoE**: 언어 게임 이론 기반 전문가 라우팅
- **토폴로지 인식 처리**: 텍스트의 공간적 구조 모델링

## 🚀 빠른 시작

### 설치

```bash
git clone https://github.com/yourusername/holographic-llm.git
cd holographic-llm
pip install -r requirements.txt
```

### 텍스트 생성

```bash
# 대화형 생성
python scripts/generate.py --checkpoint outputs/best_model.pt --interactive

# 단일 프롬프트
python scripts/generate.py --checkpoint outputs/best_model.pt --prompt "옛날 옛적에"
```

### 모델 학습

```bash
python train_main.py
```

## 🏗️ 아키텍처

### 홀로그래픽 어텐션 메커니즘

```python
# 단순화된 개념
def holographic_attention(Q, K, V):
    # 주파수 도메인으로 변환
    Q_freq = torch.fft.rfft(Q, dim=-1)
    K_freq = torch.fft.rfft(K, dim=-1)
    
    # 간섭 패턴 생성
    interference = Q_freq * K_freq.conj()
    
    # 위상 정보와 함께 값에 적용
    output_freq = interference * torch.fft.rfft(V, dim=-1)
    
    # 공간 도메인으로 복귀
    output = torch.fft.irfft(output_freq, dim=-1)
    
    return output
```

### 모델 사양

- **파라미터**: 2억 2720만개
- **레이어**: 12개 트랜스포머 블록
- **히든 차원**: 768
- **어텐션 헤드**: 12개
- **어휘 크기**: 32,000 (LLaMA 토크나이저)
- **컨텍스트 길이**: 512 토큰

## 📊 현재 상태

### ✅ 작동하는 기능

- 영어/한글 기본 텍스트 생성
- FFT 기반 어텐션 연산
- 표준 어텐션과 안전한 블렌딩 (폴백 메커니즘)
- 콘텐츠 카테고리 기반 동적 전문가 라우팅
- 메모리 효율을 위한 그래디언트 체크포인팅

### ⚠️ 알려진 한계

- **생성 품질**: 반복 패턴 발생, 특히 숫자
- **학습 데이터**: 제한된 데이터셋으로 인한 과적합
- **계산 효율**: 프로덕션 사용 최적화 필요
- **한글 성능**: 더 많은 한글 학습 데이터 필요

### 📈 학습 진행 상황

| 체크포인트 | 손실값 | 상태 |
|-----------|--------|------|
| 300 | 3.90 | 최적 |
| 400 | 4.31 | 과적합 시작 |

## 🔬 연구 방향

### 단기 목표

1. **곡률 기반 어텐션**: 곡률 공간에서 데이터 "돌돌 말기" 압축
2. **2D/3D 홀로그래픽 패턴**: 1D 근사를 넘어선 확장
3. **대규모 학습**: 위키피디아 규모 데이터셋
4. **수학적 정식화**: 엄밀한 이론적 기반 구축

### 장기 비전

- 홀로그래픽 연산을 위한 하드웨어 가속
- 비전 모델과의 통합 (진정한 홀로그래픽 처리)
- 간섭 패턴 기반 새로운 압축 알고리즘
- 크로스 모달 홀로그래픽 표현

## 🤝 기여하기

연구자, 엔지니어, 그리고 호기심 많은 분들의 기여를 환영합니다!

### 도움이 필요한 영역

1. **생성 품질**: 반복 문제 해결
2. **수학 이론**: 홀로그래픽 어텐션 정식화
3. **데이터 파이프라인**: 대규모 데이터셋 준비
4. **벤치마킹**: 표준 트랜스포머와 성능 비교
5. **문서화**: 튜토리얼 및 설명 작성

### 기여 방법

1. 저장소 포크
2. 피처 브랜치 생성 (`git checkout -b feature/놀라운기능`)
3. 변경사항 커밋 (`git commit -m '놀라운 기능 추가'`)
4. 브랜치에 푸시 (`git push origin feature/놀라운기능`)
5. Pull Request 열기

## 📚 프로젝트 구조

```
holographic-llm-v2/
├── models/              # 핵심 모델 구현
│   ├── attention.py     # 홀로그래픽 어텐션 메커니즘
│   ├── moe_ffn.py      # 동적 카테고리 MoE
│   └── topology.py      # 토폴로지 프로세서
├── training/            # 학습 로직
├── configs/             # 설정 파일
├── data/               # 데이터셋 유틸리티
└── scripts/            # 생성 및 유틸리티 스크립트
```

## 📖 배경 이야기

이 프로젝트는 단순한 챗봇 실험에서 시작해 물리학 원리(홀로그래피)가 AI 아키텍처를 어떻게 향상시킬 수 있는지 탐구하는 여정으로 발전했습니다. 3개월에 걸쳐 기본 텍스트 생성에서 파동 간섭 패턴에서 영감을 받은 새로운 어텐션 메커니즘 구현으로 변모했습니다.

"홀로그래픽 LLM"이라는 이름은 핵심 아이디어를 반영합니다: 홀로그램이 2D 간섭 패턴에 3D 정보를 인코딩하듯, 이 모델은 주파수 도메인 간섭을 통해 복잡한 언어적 관계를 인코딩하려 시도합니다.

## 🎯 프로젝트 목표

### 왜 홀로그래픽 어텐션인가?

1. **메모리 효율성**: O(N²) → O(N log N) 복잡도 개선 가능성
2. **정보 밀도**: 간섭 패턴으로 더 많은 정보 압축
3. **물리학적 직관**: 자연 현상에서 영감받은 AI
4. **새로운 가능성**: 기존 방법론의 한계 돌파 시도

## 📝 인용

이 코드를 연구에 사용하신다면 다음과 같이 인용해 주세요:

```bibtex
@software{holographic_llm_2024,
  author = {우진현},
  title = {홀로그래픽 LLM: FFT 기반 어텐션 메커니즘},
  year = {2024},
  url = {https://github.com/yourusername/holographic-llm}
}
```

## 📬 연락처

**우진현 (Jinhyun Woo)**
- 이메일: ggunio5782@gmail.com
- LinkedIn: [www.linkedin.com/in/namuneup](www.linkedin.com/in/namuneup)

다음과 같은 경우 언제든 연락주세요:
- 구현에 대한 질문
- 협업 기회
- 연구 토론
- 또는 그냥 "재밌네요!"라고 말하고 싶으실 때

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- 물리학의 홀로그래피 원리에서 영감
- 트랜스포머 아키텍처 기반 구축
- Meta AI의 LLaMA 토크나이저
- PyTorch 및 Hugging Face 커뮤니티

---

**참고**: 이것은 실험적 연구 프로젝트입니다. 새로운 개념을 보여주지만 프로덕션 사용 준비는 되지 않았습니다. 탐구, 실험, 개선을 권장합니다!

*"미래를 예측하는 가장 좋은 방법은 그것을 발명하는 것이다."* - 앨런 케이