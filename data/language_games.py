"""
동적 범주별 언어 게임 데이터 생성기
홀로그래픽 LLM을 위한 구조화된 대화 데이터
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime
import uuid

from .schema import (
    HolographicDataSample, ContentData, TopologyInfo, 
    GeometryInfo, DynamicCategory, HolographicPattern
)

class LanguageGameGenerator:
    """언어 게임별 데이터 생성기"""
    
    def __init__(self):
        # 개념 네트워크 (위상 정보 구성용)
        self.concept_graph = self._build_concept_graph()
        
        # 템플릿 저장소
        self.game_templates = {
            "explanation": self._explanation_templates(),
            "reasoning": self._reasoning_templates(), 
            "creative": self._creative_templates(),
            "analysis": self._analysis_templates(),
            "dialogue": self._dialogue_templates()
        }
        
        # 복잡도별 어휘
        self.vocabulary_levels = self._build_vocabulary_levels()
    
    def _build_concept_graph(self) -> Dict[str, Dict[str, Any]]:
        """개념 그래프 구축 - 위상정보 기반"""
        return {
            # 과학 클러스터
            "science": {
                "children": ["physics", "chemistry", "biology", "mathematics"],
                "level": 0,
                "coords": [0.0, 0.0, 1.0]
            },
            "physics": {
                "parent": "science",
                "children": ["quantum", "relativity", "thermodynamics"],
                "neighbors": ["mathematics", "chemistry"],
                "level": 1,
                "coords": [0.2, 0.1, 0.9]
            },
            "quantum": {
                "parent": "physics", 
                "neighbors": ["wave", "particle", "superposition"],
                "level": 2,
                "coords": [0.3, 0.2, 0.8]
            },
            
            # 철학 클러스터  
            "philosophy": {
                "children": ["ethics", "logic", "metaphysics", "epistemology"],
                "level": 0,
                "coords": [-1.0, 0.0, 1.0]
            },
            "consciousness": {
                "parent": "philosophy",
                "neighbors": ["mind", "awareness", "experience"],
                "level": 2,
                "coords": [-0.8, 0.3, 0.7]
            },
            
            # 기술 클러스터
            "technology": {
                "children": ["ai", "computing", "internet", "robotics"],
                "level": 0,
                "coords": [1.0, -1.0, 0.5]
            },
            "ai": {
                "parent": "technology",
                "children": ["ml", "nlp", "cv", "rl"],
                "neighbors": ["computing", "mathematics"],
                "level": 1,
                "coords": [0.9, -0.8, 0.6]
            },
            "holographic": {
                "parent": "ai",
                "neighbors": ["quantum", "interference", "attention"],
                "level": 3,
                "coords": [0.95, -0.75, 0.8]
            }
        }
    
    def _explanation_templates(self) -> List[Dict[str, Any]]:
        """설명 게임 템플릿"""
        return [
            {
                "pattern": "What is {concept}?",
                "response_pattern": "{concept} is {definition}. It works by {mechanism}.",
                "complexity": 2,
                "abstraction": 3
            },
            {
                "pattern": "How does {concept} relate to {related_concept}?", 
                "response_pattern": "{concept} and {related_concept} are connected through {relationship}.",
                "complexity": 3,
                "abstraction": 4
            },
            {
                "pattern": "Can you explain {concept} using an analogy?",
                "response_pattern": "Think of {concept} like {analogy}. Just as {analogy_detail}, {concept} {concept_detail}.",
                "complexity": 4,
                "abstraction": 3
            }
        ]
    
    def _reasoning_templates(self) -> List[Dict[str, Any]]:
        """추론 게임 템플릿"""
        return [
            {
                "pattern": "If {premise1} and {premise2}, then what can we conclude?",
                "response_pattern": "Given that {premise1} and {premise2}, we can logically conclude that {conclusion}.",
                "complexity": 4,
                "abstraction": 4
            },
            {
                "pattern": "What are the implications of {concept}?",
                "response_pattern": "The implications of {concept} include {implication1}, {implication2}, and {implication3}.",
                "complexity": 3,
                "abstraction": 4
            },
            {
                "pattern": "Why might {concept1} lead to {concept2}?",
                "response_pattern": "{concept1} might lead to {concept2} because {causal_mechanism}.",
                "complexity": 3,
                "abstraction": 3
            }
        ]
    
    def _creative_templates(self) -> List[Dict[str, Any]]:
        """창작 게임 템플릿"""
        return [
            {
                "pattern": "Imagine a world where {concept} works differently. How?",
                "response_pattern": "In this world, {concept} would {alternative_mechanism}, leading to {consequences}.",
                "complexity": 4,
                "abstraction": 5
            },
            {
                "pattern": "Create a metaphor for {concept}.",
                "response_pattern": "{concept} is like {metaphor_base} - {metaphor_explanation}.",
                "complexity": 3,
                "abstraction": 4
            },
            {
                "pattern": "What if we combined {concept1} with {concept2}?",
                "response_pattern": "Combining {concept1} with {concept2} might create {hybrid_concept} that {novel_properties}.",
                "complexity": 5,
                "abstraction": 5
            }
        ]
    
    def _analysis_templates(self) -> List[Dict[str, Any]]:
        """분석 게임 템플릿"""
        return [
            {
                "pattern": "Break down {concept} into its components.",
                "response_pattern": "{concept} consists of {component1}, {component2}, and {component3}.",
                "complexity": 3,
                "abstraction": 2
            },
            {
                "pattern": "What patterns do you see in {concept}?",
                "response_pattern": "I observe these patterns in {concept}: {pattern1}, {pattern2}, {pattern3}.",
                "complexity": 4,
                "abstraction": 3
            },
            {
                "pattern": "Compare and contrast {concept1} and {concept2}.",
                "response_pattern": "{concept1} and {concept2} are similar in {similarity} but differ in {difference}.",
                "complexity": 3,
                "abstraction": 3
            }
        ]
    
    def _dialogue_templates(self) -> List[Dict[str, Any]]:
        """대화 게임 템플릿"""
        return [
            {
                "pattern": "I think {opinion}. What do you think?",
                "response_pattern": "That's interesting. I {agreement_level} because {reasoning}.",
                "complexity": 2,
                "abstraction": 2
            },
            {
                "pattern": "Can you help me understand {concept}?",
                "response_pattern": "Of course! Let me explain {concept} step by step: {step1}, {step2}, {step3}.",
                "complexity": 2,
                "abstraction": 2
            },
            {
                "pattern": "What questions should I ask about {concept}?",
                "response_pattern": "Good questions about {concept} include: {question1}, {question2}, {question3}.",
                "complexity": 3,
                "abstraction": 3
            }
        ]
    
    def _build_vocabulary_levels(self) -> Dict[int, List[str]]:
        """복잡도별 어휘 구성"""
        return {
            1: ["simple", "basic", "easy", "clear", "direct"],
            2: ["understand", "explain", "connect", "related", "important"],
            3: ["analyze", "complex", "structure", "mechanism", "relationship"],
            4: ["sophisticated", "intricate", "multifaceted", "paradigm", "synthesis"],
            5: ["paradigmatic", "metamorphic", "transcendental", "dialectical", "phenomenological"]
        }
    
    def generate_sample(self, game_type: str, concept: str = None) -> HolographicDataSample:
        """단일 샘플 생성"""
        if concept is None:
            concept = random.choice(list(self.concept_graph.keys()))
        
        template = random.choice(self.game_templates[game_type])
        
        # 관련 개념들 선택
        concept_info = self.concept_graph.get(concept, {})
        related_concepts = concept_info.get("neighbors", [])
        if not related_concepts:
            related_concepts = [c for c in self.concept_graph.keys() if c != concept][:3]
        
        # 텍스트 생성
        question, response = self._generate_text_pair(template, concept, related_concepts)
        
        # 메타데이터 구성
        sample_id = str(uuid.uuid4())[:8]
        
        # Content 데이터
        content = ContentData(
            text=f"{question} {response}",
            tokens=self._tokenize(f"{question} {response}"),
            concepts=[concept] + related_concepts[:2],
            speaker="assistant",
            turn_type="qa_pair", 
            dialogue_act="explain",
            context_window=[question, response]
        )
        
        # Topology 정보
        topology = TopologyInfo(
            node_id=concept,
            neighbors=related_concepts,
            distances={rc: random.uniform(0.1, 0.9) for rc in related_concepts},
            hierarchy_level=concept_info.get("level", 1),
            parent_concepts=concept_info.get("parent", []) if isinstance(concept_info.get("parent"), list) else [concept_info.get("parent")] if concept_info.get("parent") else [],
            child_concepts=concept_info.get("children", []),
            cluster_id=self._get_cluster_id(concept),
            connectivity_score=len(related_concepts) / 10.0
        )
        
        # Geometry 정보 
        geometry = GeometryInfo(
            manifold_coords=concept_info.get("coords", [random.uniform(-1, 1) for _ in range(3)]),
            local_dimension=len(related_concepts),
            smoothness_score=random.uniform(0.3, 0.9)
        )
        
        # Dynamic Category
        dynamics = DynamicCategory(
            game_type=game_type,
            current_state=f"{game_type}_active",
            possible_transitions=self._get_possible_transitions(game_type),
            abstraction_level=template["abstraction"],
            complexity_score=template["complexity"] / 5.0,
            cognitive_load=random.uniform(0.2, 0.8)
        )
        
        # Holographic Pattern
        text_length = len(content.text)
        holographic = HolographicPattern(
            frequency_components=self._generate_frequency_components(text_length),
            dominant_frequencies=[1, 3, 7, 13],  # 피보나치 기반
            interference_strength=random.uniform(0.3, 0.8),
            coherence_length=min(text_length // 4, 50),
            local_patterns=self._extract_local_patterns(content.text),
            global_patterns=[f"{game_type}_structure", f"{concept}_domain"]
        )
        
        return HolographicDataSample(
            id=sample_id,
            version="1.0",
            created_at=datetime.now().isoformat(),
            content=content,
            topology=topology,
            geometry=geometry,
            dynamics=dynamics,
            holographic=holographic,
            targets={"next_token": response.split()[-1] if response else ""}
        )
    
    def _generate_text_pair(self, template: Dict, concept: str, related_concepts: List[str]) -> Tuple[str, str]:
        """템플릿 기반 질문-답변 쌍 생성"""
        # 개념별 설명 데이터
        explanations = {
            "holographic": "a method that uses interference patterns to capture and process information",
            "quantum": "the physics of matter and energy at the smallest scales",
            "consciousness": "the state of being aware and able to think",
            "ai": "artificial intelligence systems that can learn and reason",
            "physics": "the science of matter, energy, and their interactions"
        }
        
        mechanisms = {
            "holographic": "creating interference patterns between wave functions",
            "quantum": "probabilistic superposition and measurement collapse", 
            "consciousness": "integrating sensory information into unified experience",
            "ai": "processing data through neural networks and algorithms",
            "physics": "mathematical modeling of natural phenomena"
        }
        
        # 템플릿 변수 준비
        format_vars = {
            'concept': concept,
            'definition': explanations.get(concept, f"a concept in the domain of {self._get_cluster_id(concept)}"),
            'mechanism': mechanisms.get(concept, "structured information processing"),
            'related_concept': related_concepts[0] if related_concepts else "information",
            'relationship': f"shared mathematical foundations and {random.choice(['causal', 'structural', 'functional'])} connections",
            'step1': "first understand the basic principles",
            'step2': "then explore the applications", 
            'step3': "finally consider the implications",
            'question1': f"What are the key components of {concept}?",
            'question2': f"How does {concept} work in practice?",
            'question3': f"What are the limitations of {concept}?",
            'agreement_level': random.choice(["agree with that", "see it differently", "partially agree"]),
            'reasoning': f"the evidence suggests {concept} works through {random.choice(['systematic', 'emergent', 'structured'])} processes",
            'opinion': f"{concept} is fundamental to understanding modern systems",
            'premise1': f"{concept} demonstrates consistent patterns",
            'premise2': f"these patterns show {random.choice(['emergent', 'systematic', 'coherent'])} behavior",
            'conclusion': f"{concept} operates through well-defined principles",
            'concept1': concept,
            'concept2': related_concepts[0] if related_concepts else "information",
            'analogy': random.choice(["a musical orchestra", "a complex ecosystem", "a well-designed machine"]),
            'analogy_detail': f"each part contributes to the whole",
            'concept_detail': f"each component enhances the overall function",
            'implication1': f"broader understanding of {self._get_cluster_id(concept)}",
            'implication2': f"new approaches to problem-solving",
            'implication3': f"deeper insights into system behavior"
        }
        
        # 안전한 포매팅 - Question
        try:
            question = template["pattern"].format(**format_vars)
        except KeyError as e:
            question = f"What can you tell me about {concept}?"
        
        # 안전한 포매팅 - Response
        try:
            response = template["response_pattern"].format(**format_vars)
        except KeyError as e:
            # 누락된 변수가 있으면 기본 응답 생성
            response = f"Let me explain {concept}. It's {format_vars['definition']} and works by {format_vars['mechanism']}."
        
        return question, response
    
    def _tokenize(self, text: str) -> List[str]:
        """간단한 토큰화"""
        return text.lower().replace("?", " ?").replace(".", " .").split()
    
    def _get_cluster_id(self, concept: str) -> str:
        """개념의 클러스터 ID 반환"""
        for cluster, info in self.concept_graph.items():
            if concept in info.get("children", []) or concept == cluster:
                return cluster
        return "general"
    
    def _get_possible_transitions(self, game_type: str) -> Dict[str, float]:
        """게임 타입별 가능한 전이"""
        transitions = {
            "explanation": {"reasoning": 0.4, "analysis": 0.3, "dialogue": 0.3},
            "reasoning": {"analysis": 0.4, "creative": 0.3, "explanation": 0.3},
            "creative": {"dialogue": 0.4, "explanation": 0.3, "reasoning": 0.3},
            "analysis": {"reasoning": 0.4, "explanation": 0.4, "creative": 0.2},
            "dialogue": {"explanation": 0.3, "reasoning": 0.3, "creative": 0.4}
        }
        return transitions.get(game_type, {})
    
    def _generate_frequency_components(self, length: int) -> List[complex]:
        """주파수 성분 생성 (홀로그래픽 어텐션용)"""
        n_components = min(length // 8, 16)
        components = []
        for i in range(n_components):
            amplitude = 1.0 / (i + 1)  # 1/f 스펙트럼
            phase = random.uniform(0, 2 * np.pi)
            components.append(amplitude * np.exp(1j * phase))
        return components
    
    def _extract_local_patterns(self, text: str) -> List[str]:
        """지역적 패턴 추출"""
        words = text.lower().split()
        patterns = []
        
        # n-gram 패턴
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            if len(bigram) > 5:  # 의미있는 패턴만
                patterns.append(bigram)
        
        # 길이별 패턴
        if len(words) < 10:
            patterns.append("short_form")
        elif len(words) > 30:
            patterns.append("long_form")
        else:
            patterns.append("medium_form")
        
        return patterns[:5]  # 최대 5개
    
    def generate_dataset(self, total_samples: int = 5000, 
                        game_distribution: Dict[str, float] = None) -> List[HolographicDataSample]:
        """전체 데이터셋 생성"""
        if game_distribution is None:
            game_distribution = {
                "explanation": 0.25,
                "reasoning": 0.25, 
                "creative": 0.2,
                "analysis": 0.15,
                "dialogue": 0.15
            }
        
        samples = []
        concepts = list(self.concept_graph.keys())
        
        for game_type, proportion in game_distribution.items():
            n_samples = int(total_samples * proportion)
            
            for _ in range(n_samples):
                concept = random.choice(concepts)
                sample = self.generate_sample(game_type, concept)
                samples.append(sample)
        
        return samples