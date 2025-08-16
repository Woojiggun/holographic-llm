
"""
홀로그래픽 데이터셋 샘플 탐색기
"""
import sys
sys.path.append('.')
import json
from data.schema import HolographicDataSample

def explore_sample(sample_id=None):
    """특정 샘플 탐색"""
    with open('data/train.jsonl', 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    if sample_id:
        sample_data = next((s for s in samples if s['id'] == sample_id), None)
        if not sample_data:
            print(f"Sample {sample_id} not found!")
            return
        samples = [sample_data]
    else:
        # 첫 번째 샘플 표시
        samples = samples[:1]
    
    for sample_data in samples:
        sample = HolographicDataSample.from_dict(sample_data)
        
        print(f"\n{'='*50}")
        print(f"Sample ID: {sample.id}")
        print(f"{'='*50}")
        
        print(f"\n[Content]")
        print(f"Text: {sample.content.text}")
        print(f"Concepts: {sample.content.concepts}")
        print(f"Speaker: {sample.content.speaker}")
        print(f"Turn Type: {sample.content.turn_type}")
        
        print(f"\n[Dynamics]")
        print(f"Game Type: {sample.dynamics.game_type}")  
        print(f"Abstraction Level: {sample.dynamics.abstraction_level}")
        print(f"Complexity Score: {sample.dynamics.complexity_score:.2f}")
        print(f"Cognitive Load: {sample.dynamics.cognitive_load:.2f}")
        
        print(f"\n[Topology]")
        print(f"Node ID: {sample.topology.node_id}")
        print(f"Neighbors: {sample.topology.neighbors}")
        print(f"Hierarchy Level: {sample.topology.hierarchy_level}")
        print(f"Connectivity: {sample.topology.connectivity_score:.2f}")
        
        print(f"\n[Geometry]")  
        print(f"Manifold Coords: {sample.geometry.manifold_coords}")
        print(f"Local Dimension: {sample.geometry.local_dimension}")
        print(f"Smoothness: {sample.geometry.smoothness_score:.2f}")
        
        print(f"\n[Holographic]")
        print(f"Dominant Frequencies: {sample.holographic.dominant_frequencies}")
        print(f"Interference Strength: {sample.holographic.interference_strength:.2f}")
        print(f"Coherence Length: {sample.holographic.coherence_length}")
        print(f"Local Patterns: {sample.holographic.local_patterns}")

if __name__ == "__main__":
    import sys
    sample_id = sys.argv[1] if len(sys.argv) > 1 else None
    explore_sample(sample_id)
    