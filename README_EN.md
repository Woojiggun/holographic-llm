# Holographic LLM: FFT-based Attention Mechanism Experiment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> An experimental language model exploring holographic principles in attention mechanisms through FFT-based interference patterns.

## 🌟 Key Innovation

This project introduces **Holographic Attention** - a novel approach that applies physical holography concepts to transformer architectures:

- **FFT-based Interference Patterns**: Processes attention in frequency domain
- **Phase Information Encoding**: Utilizes complex phase for richer representations
- **Dynamic Category MoE**: Language game-inspired expert routing
- **Topology-aware Processing**: Spatial structure modeling in text

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/holographic-llm.git
cd holographic-llm
pip install -r requirements.txt
```

### Generate Text

```bash
# Interactive generation
python scripts/generate.py --checkpoint outputs/best_model.pt --interactive

# Single prompt
python scripts/generate.py --checkpoint outputs/best_model.pt --prompt "Once upon a time"
```

### Train Your Own Model

```bash
python train_main.py
```

## 🏗️ Architecture

### Holographic Attention Mechanism

```python
# Simplified concept
def holographic_attention(Q, K, V):
    # Transform to frequency domain
    Q_freq = torch.fft.rfft(Q, dim=-1)
    K_freq = torch.fft.rfft(K, dim=-1)
    
    # Create interference pattern
    interference = Q_freq * K_freq.conj()
    
    # Apply to values with phase information
    output_freq = interference * torch.fft.rfft(V, dim=-1)
    
    # Back to spatial domain
    output = torch.fft.irfft(output_freq, dim=-1)
    
    return output
```

### Model Specifications

- **Parameters**: 227.2M
- **Layers**: 12 transformer blocks
- **Hidden Dimension**: 768
- **Attention Heads**: 12
- **Vocabulary**: 32,000 (LLaMA tokenizer)
- **Context Length**: 512 tokens

## 📊 Current Status

### ✅ What Works

- Basic text generation in English and Korean
- FFT-based attention computation
- Safe blending with standard attention (fallback mechanism)
- Dynamic expert routing based on content categories
- Gradient checkpointing for memory efficiency

### ⚠️ Known Limitations

- **Generation Quality**: Repetitive patterns, especially with numbers
- **Training Data**: Limited dataset leads to overfitting
- **Computational**: Not optimized for production use
- **Korean Performance**: Needs more Korean training data

### 📈 Training Progress

| Checkpoint | Loss | Status |
|------------|------|--------|
| 300 | 3.90 | Best |
| 400 | 4.31 | Overfitting begins |

## 🔬 Research Directions

### Near-term Goals

1. **Curvature-based Attention**: Compress data by "rolling it up" in curved space
2. **2D/3D Holographic Patterns**: Extend beyond 1D approximation
3. **Large-scale Training**: Wikipedia-scale datasets
4. **Mathematical Formalization**: Rigorous theoretical foundation

### Long-term Vision

- Hardware acceleration for holographic operations
- Integration with vision models (true holographic processing)
- Novel compression algorithms based on interference patterns
- Cross-modal holographic representations

## 🤝 Contributing

We welcome contributions from researchers, engineers, and enthusiasts!

### Areas Needing Help

1. **Generation Quality**: Solving repetition issues
2. **Mathematical Theory**: Formalizing holographic attention
3. **Data Pipeline**: Large-scale dataset preparation
4. **Benchmarking**: Performance comparison with standard transformers
5. **Documentation**: Tutorials and explanations

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Project Structure

```
holographic-llm-v2/
├── models/              # Core model implementations
│   ├── attention.py     # Holographic attention mechanism
│   ├── moe_ffn.py      # Dynamic category MoE
│   └── topology.py      # Topology processor
├── training/            # Training logic
├── configs/             # Configuration files
├── data/               # Dataset utilities
└── scripts/            # Generation and utility scripts
```

## 📖 Background Story

This project started as a simple chatbot experiment and evolved into an exploration of how physical principles (holography) might enhance AI architectures. Over 3 months, it transformed from basic text generation to implementing novel attention mechanisms inspired by wave interference patterns.

The name "Holographic LLM" reflects the core idea: just as holograms encode 3D information in 2D interference patterns, this model attempts to encode complex linguistic relationships through frequency-domain interference.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{holographic_llm_2024,
  author = {Woo, Jinhyun},
  title = {Holographic LLM: FFT-based Attention Mechanism},
  year = {2024},
  url = {https://github.com/yourusername/holographic-llm}
}
```

## 📬 Contact

**Jinhyun Woo**
- Email: ggunio5782@gmail.com
- LinkedIn: [www.linkedin.com/in/namuneup](www.linkedin.com/in/namuneup)

Feel free to reach out for:
- Questions about the implementation
- Collaboration opportunities
- Research discussions
- Or just to say "this is interesting!"

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by holographic principles in physics
- Built upon the transformer architecture
- LLaMA tokenizer from Meta AI
- PyTorch and Hugging Face communities

---

**Note**: This is an experimental research project. While it demonstrates novel concepts, it's not ready for production use. We encourage exploration, experimentation, and improvement!

*"The best way to predict the future is to invent it."* - Alan Kay