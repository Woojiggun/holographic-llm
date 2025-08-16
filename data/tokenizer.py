"""
개선된 토크나이저 - 한글 지원 강화
"""
from transformers import AutoTokenizer
import torch
import warnings

class ImprovedTokenizerManager:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.tokenizer_type = None
        
        # 한글 지원 토크나이저들을 우선순위대로 시도
        tokenizer_candidates = [
            # 1. 한국어 특화 토크나이저들
            ("klue/bert-base", "KLUE BERT (Korean optimized)"),
            ("monologg/kobert", "KoBERT (Korean BERT)"),
            
            # 2. 다국어 지원 토크나이저들  
            ("microsoft/DialoGPT-medium", "DialoGPT (Multilingual)"),
            ("facebook/blenderbot-400M-distill", "BlenderBot (Multilingual)"),
            
            # 3. 기본 다국어 모델들
            ("bert-base-multilingual-cased", "BERT Multilingual"),
            ("xlm-roberta-base", "XLM-RoBERTa"),
            
            # 4. 폴백 옵션
            ("huggyllama/llama-7b", "LLaMA (fallback)"),
        ]
        
        self.tokenizer, self.tokenizer_type = self._load_best_tokenizer(tokenizer_candidates)
        
        # 특수 토큰 설정
        self._setup_special_tokens()
        
        # Vocab 크기 자동 조정
        self._adjust_vocab_size()
        
        # 한글 토큰화 품질 테스트
        self._test_korean_quality()
    
    def _load_best_tokenizer(self, candidates):
        """가장 적합한 토크나이저 로드"""
        for model_name, description in candidates:
            try:
                print(f"Trying {description} ({model_name})...")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_length=self.config.max_seq_length,
                    padding_side="right",
                    use_fast=True,
                    trust_remote_code=True
                )
                
                # 한글 테스트
                test_text = "안녕하세요 Hello"
                tokens = tokenizer.tokenize(test_text)
                decoded = tokenizer.decode(tokenizer.encode(test_text), skip_special_tokens=True)
                
                # 한글이 제대로 처리되는지 확인
                if "안녕하세요" in decoded and "Hello" in decoded:
                    print(f"Successfully loaded: {description}")
                    return tokenizer, description
                else:
                    print(f"Korean support test failed for {description}")
                    
            except Exception as e:
                print(f"Failed to load {description}: {e}")
                continue
        
        raise RuntimeError("No suitable tokenizer found with Korean support")
    
    def _setup_special_tokens(self):
        """특수 토큰 설정"""
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # 새로운 패딩 토큰 추가
                special_tokens = {"pad_token": "<pad>"}
                self.tokenizer.add_special_tokens(special_tokens)
        
        # BOS 토큰 설정 (없는 경우)
        if self.tokenizer.bos_token is None:
            if hasattr(self.tokenizer, 'cls_token') and self.tokenizer.cls_token:
                self.tokenizer.bos_token = self.tokenizer.cls_token
        
        # 토큰 ID들 설정
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = getattr(self.tokenizer, 'bos_token_id', self.tokenizer.cls_token_id if hasattr(self.tokenizer, 'cls_token_id') else None)
    
    def _adjust_vocab_size(self):
        """Vocab 크기 자동 조정"""
        actual_vocab_size = len(self.tokenizer)
        if actual_vocab_size != self.config.vocab_size:
            print(f"Auto-adjusting vocab_size: {self.config.vocab_size} -> {actual_vocab_size}")
            self.config.vocab_size = actual_vocab_size
    
    def _test_korean_quality(self):
        """한글 토큰화 품질 테스트"""
        print(f"\n한글 토큰화 품질 테스트:")
        
        test_cases = [
            "안녕하세요",
            "한국어",
            "홀로그래픽",
            "과학기술",
            "Hello 안녕",
            "AI는 인공지능입니다"
        ]
        
        total_score = 0
        for test_text in test_cases:
            tokens = self.tokenizer.tokenize(test_text)
            encoded = self.tokenizer.encode(test_text, add_special_tokens=False)
            decoded = self.tokenizer.decode(encoded, skip_special_tokens=True)
            
            # 정확도 계산
            accuracy = 1.0 if decoded.strip() == test_text.strip() else 0.0
            total_score += accuracy
            
            print(f"  '{test_text}' -> {len(tokens)} tokens -> '{decoded}' ({'OK' if accuracy else 'FAIL'})")
        
        avg_score = total_score / len(test_cases)
        print(f"한글 처리 정확도: {avg_score*100:.1f}%")
        
        if avg_score < 0.8:
            warnings.warn("한글 토큰화 품질이 낮습니다. 다른 토크나이저를 고려해보세요.")
    
    def encode_with_validation(self, text, **kwargs):
        """검증이 포함된 인코딩"""
        try:
            result = self.tokenizer(text, **kwargs)
            
            # 디코딩 테스트
            if 'input_ids' in result:
                decoded = self.tokenizer.decode(result['input_ids'], skip_special_tokens=True)
                # 기본적인 정확성 체크
                if len(decoded.strip()) == 0 and len(text.strip()) > 0:
                    warnings.warn(f"Tokenization may have failed for: '{text}'")
            
            return result
        except Exception as e:
            print(f"Encoding error for '{text}': {e}")
            # 폴백: 간단한 인코딩
            return self.tokenizer(text, add_special_tokens=True, return_tensors='pt')
    
    def decode_with_cleanup(self, token_ids, **kwargs):
        """정리가 포함된 디코딩"""
        try:
            # 기본 디코딩
            decoded = self.tokenizer.decode(token_ids, **kwargs)
            
            # 후처리: 불필요한 공백 제거
            decoded = decoded.strip()
            
            # 한글 관련 정리 (필요시)
            # decoded = decoded.replace('  ', ' ')  # 이중 공백 제거
            
            return decoded
            
        except Exception as e:
            print(f"Decoding error: {e}")
            # 폴백: 토큰별 개별 디코딩
            try:
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                tokens = [self.tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
                return ''.join(tokens)
            except:
                return "[Decode Error]"

# 호환성을 위한 별칭
TokenizerManager = ImprovedTokenizerManager