"""
미래형 데이터셋 스키마 - Forward Compatible Design
홀로그래픽 LLM, 동적 범주, 위상정보, 곡률 벡터까지 지원
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import numpy as np

@dataclass
class TopologyInfo:
    """위상 정보 - 개념 간 관계와 구조"""
    # 그래프 구조
    node_id: str
    neighbors: List[str] = field(default_factory=list)
    distances: Dict[str, float] = field(default_factory=dict)
    
    # 계층 구조  
    hierarchy_level: int = 0
    parent_concepts: List[str] = field(default_factory=list)
    child_concepts: List[str] = field(default_factory=list)
    
    # 클러스터 정보
    cluster_id: str = ""
    cluster_center: List[float] = field(default_factory=list)
    connectivity_score: float = 0.0

@dataclass 
class GeometryInfo:
    """기하학적 정보 - 곡률 벡터 등 미래 확장"""
    # 매니폴드 좌표
    manifold_coords: List[float] = field(default_factory=list)
    
    # 곡률 정보 (미래 확장)
    curvature_vector: Optional[List[float]] = None
    tangent_space: Optional[List[List[float]]] = None
    metric_tensor: Optional[List[List[float]]] = None
    
    # 국소 기하 속성
    local_dimension: int = 0
    smoothness_score: float = 0.0

@dataclass
class DynamicCategory:
    """동적 범주 정보 - 언어 게임별 분류"""
    # 언어 게임 유형
    game_type: str  # "explanation", "reasoning", "creative", "analysis", "dialogue"
    
    # 범주 전이 정보
    current_state: str
    possible_transitions: Dict[str, float] = field(default_factory=dict)
    transition_history: List[str] = field(default_factory=list)
    
    # 인지 속성
    abstraction_level: int = 1  # 1=구체적, 5=추상적
    complexity_score: float = 0.0
    cognitive_load: float = 0.0

@dataclass
class HolographicPattern:
    """홀로그래픽 패턴 정보 - FFT 어텐션 최적화"""
    # 주파수 도메인 특성
    frequency_components: List[complex] = field(default_factory=list)
    dominant_frequencies: List[int] = field(default_factory=list)
    
    # 간섭 패턴 정보
    interference_strength: float = 0.0
    coherence_length: int = 0
    phase_shifts: List[float] = field(default_factory=list)
    
    # 다중 스케일 구조
    local_patterns: List[str] = field(default_factory=list)
    global_patterns: List[str] = field(default_factory=list)
    scale_hierarchy: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentData:
    """콘텐츠 데이터 - 다양한 모달리티 지원"""
    # 기본 텍스트
    text: str
    
    # 구조화된 정보
    tokens: List[str] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list)
    syntactic_tree: Optional[Dict[str, Any]] = None
    
    # 의미 정보
    semantic_roles: Dict[str, str] = field(default_factory=dict)
    named_entities: List[Dict[str, str]] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    
    # 대화 정보
    speaker: str = "system"
    turn_type: str = "statement"  # statement, question, response
    dialogue_act: str = "inform"
    context_window: List[str] = field(default_factory=list)

@dataclass
class HolographicDataSample:
    """완전한 홀로그래픽 데이터 샘플"""
    # 핵심 데이터 (필수)
    id: str
    content: ContentData
    topology: TopologyInfo  
    geometry: GeometryInfo
    dynamics: DynamicCategory
    holographic: HolographicPattern
    
    # 메타정보 (기본값 있음)
    version: str = "1.0"
    created_at: str = ""
    targets: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        return {
            'id': self.id,
            'version': self.version, 
            'created_at': self.created_at,
            'content': {
                'text': self.content.text,
                'tokens': self.content.tokens,
                'pos_tags': self.content.pos_tags,
                'syntactic_tree': self.content.syntactic_tree,
                'semantic_roles': self.content.semantic_roles,
                'named_entities': self.content.named_entities,
                'concepts': self.content.concepts,
                'speaker': self.content.speaker,
                'turn_type': self.content.turn_type,
                'dialogue_act': self.content.dialogue_act,
                'context_window': self.content.context_window
            },
            'topology': {
                'node_id': self.topology.node_id,
                'neighbors': self.topology.neighbors,
                'distances': self.topology.distances,
                'hierarchy_level': self.topology.hierarchy_level,
                'parent_concepts': self.topology.parent_concepts,
                'child_concepts': self.topology.child_concepts,
                'cluster_id': self.topology.cluster_id,
                'cluster_center': self.topology.cluster_center,
                'connectivity_score': self.topology.connectivity_score
            },
            'geometry': {
                'manifold_coords': self.geometry.manifold_coords,
                'curvature_vector': self.geometry.curvature_vector,
                'tangent_space': self.geometry.tangent_space,
                'metric_tensor': self.geometry.metric_tensor,
                'local_dimension': self.geometry.local_dimension,
                'smoothness_score': self.geometry.smoothness_score
            },
            'dynamics': {
                'game_type': self.dynamics.game_type,
                'current_state': self.dynamics.current_state,
                'possible_transitions': self.dynamics.possible_transitions,
                'transition_history': self.dynamics.transition_history,
                'abstraction_level': self.dynamics.abstraction_level,
                'complexity_score': self.dynamics.complexity_score,
                'cognitive_load': self.dynamics.cognitive_load
            },
            'holographic': {
                'frequency_components': [{'real': c.real, 'imag': c.imag} for c in self.holographic.frequency_components],
                'dominant_frequencies': self.holographic.dominant_frequencies,
                'interference_strength': self.holographic.interference_strength,
                'coherence_length': self.holographic.coherence_length,
                'phase_shifts': self.holographic.phase_shifts,
                'local_patterns': self.holographic.local_patterns,
                'global_patterns': self.holographic.global_patterns,
                'scale_hierarchy': self.holographic.scale_hierarchy
            },
            'targets': self.targets
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HolographicDataSample':
        """딕셔너리에서 객체 생성"""
        content = ContentData(**data['content'])
        topology = TopologyInfo(**data['topology'])  
        geometry = GeometryInfo(**data['geometry'])
        dynamics = DynamicCategory(**data['dynamics'])
        
        # 복소수 복원
        freq_components = []
        if 'frequency_components' in data['holographic']:
            for c in data['holographic']['frequency_components']:
                freq_components.append(complex(c['real'], c['imag']))
        
        holographic_data = data['holographic'].copy()
        holographic_data['frequency_components'] = freq_components
        holographic = HolographicPattern(**holographic_data)
        
        return cls(
            id=data['id'],
            version=data.get('version', '1.0'),
            created_at=data.get('created_at', ''),
            content=content,
            topology=topology,
            geometry=geometry, 
            dynamics=dynamics,
            holographic=holographic,
            targets=data.get('targets', {})
        )

class HolographicDataset:
    """홀로그래픽 데이터셋 관리자"""
    
    def __init__(self, schema_version: str = "1.0"):
        self.schema_version = schema_version
        self.samples: List[HolographicDataSample] = []
        
    def add_sample(self, sample: HolographicDataSample):
        """샘플 추가"""
        self.samples.append(sample)
    
    def save_to_jsonl(self, filepath: str):
        """JSONL 형식으로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                json.dump(sample.to_dict(), f, ensure_ascii=False)
                f.write('\n')
    
    def load_from_jsonl(self, filepath: str):
        """JSONL 형식에서 로드"""
        self.samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                sample = HolographicDataSample.from_dict(data)
                self.samples.append(sample)
    
    def get_stats(self) -> Dict[str, Any]:
        """데이터셋 통계"""
        if not self.samples:
            return {}
            
        game_types = [s.dynamics.game_type for s in self.samples]
        complexity_scores = [s.dynamics.complexity_score for s in self.samples]
        
        return {
            'total_samples': len(self.samples),
            'game_type_distribution': {gt: game_types.count(gt) for gt in set(game_types)},
            'avg_complexity': np.mean(complexity_scores) if complexity_scores else 0,
            'avg_text_length': np.mean([len(s.content.text) for s in self.samples]),
            'unique_concepts': len(set([c for s in self.samples for c in s.content.concepts])),
            'schema_version': self.schema_version
        }