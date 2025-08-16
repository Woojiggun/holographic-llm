# Holographic LLM V2

혁신적인 홀로그래픽 어텐션 메커니즘을 사용한 실험적 언어 모델

## 주요 특징

### 1. 홀로그래픽 어텐션 (Holographic Attention)
- FFT 기반 주파수 도메인 처리
- 간섭 패턴을 통한 정보 인코딩
- 위상 정보를 활용한 관계 모델링

### 2. 동적 카테고리 MoE (Dynamic Category MoE)
- Language Games 기반 동적 전문가 선택
- 카테고리별 특화 처리
- 적응적 게이팅 메커니즘

### 3. 토폴로지 정보 시스템 (Topology Information)
- 공간적 정보 구조화
- k-NN 기반 지역 정보 집계
- 위치 인식 처리

## 모델 사양

- **파라미터**: 227.2M
- **레이어**: 12
- **Hidden Dim**: 768
- **Attention Heads**: 12
- **토크나이저**: LLaMA (32,000 vocab)

## 현재 상태

⚠️ **실험적 프로젝트 (Experimental)**

- ✅ 아키텍처 구현 완료
- ✅ 기본 학습 가능
- ⚠️ 영어 생성 기본 수준
- ❌ 한글 생성 개선 필요
- ⚠️ 더 많은 학습 필요 (현재 400 스텝)

## 설치

```bash
pip install torch transformers tqdm
```

## 사용법

### 학습
```bash
python train_main.py
```

### 텍스트 생성
```bash
python scripts/generate.py --checkpoint outputs/best_model.pt --prompt "Hello"
```

### 대화형 생성
```bash
python scripts/generate.py --checkpoint outputs/best_model.pt --interactive
```

## 프로젝트 구조

```
holographic-llm-v2/
├── models/
│   ├── holographic_llm.py    # 메인 모델
│   ├── attention.py           # 홀로그래픽 어텐션
│   ├── moe_ffn.py            # Dynamic Category MoE
│   └── topology.py           # 토폴로지 프로세서
├── training/
│   └── trainer.py            # 학습 로직
├── configs/
│   └── base_config.py        # 설정
├── data/
│   └── dataset.py            # 데이터 로더
└── scripts/
    └── generate.py           # 생성 스크립트
```

## Known Issues

- 한글 텍스트 생성 품질 낮음
- 반복 패턴 발생 (특히 숫자)
- 더 많은 학습 데이터 필요
- 학습 초기 단계 (400 스텝)

## 개선 방향

1. **학습 확장**
   - 최소 10,000 스텝 이상 학습
   - 더 큰 한글 데이터셋 사용
   - Learning rate scheduling 개선

2. **아키텍처 튜닝**
   - Holographic strength 조정
   - Dropout 비율 증가
   - Category 수 최적화

3. **데이터 개선**
   - 한글-영어 균형 데이터셋
   - 고품질 텍스트 필터링
   - 도메인별 특화 데이터

## 기술적 혁신

이 프로젝트는 다음과 같은 아이디어를 구현합니다:

1. **FFT 기반 어텐션**: 주파수 도메인에서의 정보 처리로 장거리 의존성 포착
2. **Language Games**: Wittgenstein의 철학적 개념을 MoE에 적용
3. **토폴로지 인식**: 텍스트의 공간적 구조를 명시적으로 모델링

## 라이선스

MIT License

## 기여

실험적 프로젝트로 개선 아이디어와 PR을 환영합니다!

## 참고

- 이 프로젝트는 연구 목적의 실험적 구현입니다
- 프로덕션 사용을 위해서는 추가 학습과 최적화가 필요합니다
- 홀로그래픽 원리를 NLP에 적용한 새로운 시도입니다