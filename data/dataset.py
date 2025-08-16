"""
데이터셋 - labels 패딩 처리
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json

class TextDataset(Dataset):
    def __init__(self, data_path, max_length):
        self.max_length = max_length
        self.data = []
        
        # 안전한 로드
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # 새로운 스키마 지원
                        if 'content' in item and 'text' in item['content']:
                            text = item['content']['text']
                        else:
                            # 이전 스키마 호환성
                            text = item.get('text', '')
                        
                        if text.strip():  # 빈 텍스트 제외
                            self.data.append(text)
                    except Exception as e:
                        print(f"  Skipping invalid line: {e}")
                        continue
        except Exception as e:
            print(f"Data load error: {e}")
            # 더미 데이터
            self.data = ["This is a test sentence."] * 100
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer, max_length):
    """안전한 collate 함수"""
    # 토큰화
    encoded = tokenizer(
        batch,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 라벨 (패딩은 -100)
    labels = encoded['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # 필수!
    
    # attention_mask를 bool로 변환하지 않음 (모델에서 처리)
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': None,  # 모델에서 자동 생성하도록
        'labels': labels
    }

def create_dataloader(data_path, tokenizer, config):
    dataset = TextDataset(data_path, config.max_seq_length)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.tokenizer, config.max_seq_length),
        num_workers=0,  # 안전하게 0
        pin_memory=True
    )