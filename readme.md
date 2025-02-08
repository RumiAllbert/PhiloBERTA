# PhiloBERTA: A Transformer-Based Cross-Lingual Analysis of Greek and Latin Lexicon

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-%23EE4C2C.svg)](https://pytorch.org/)

## Overview

PHILOBERTA is a specialized transformer-based framework designed to analyze and quantify semantic relationships between Ancient Greek and Latin philosophical terms. By leveraging modern language models and classical linguistics, our approach bridges the gap between ancient languages and contemporary NLP techniques.

The project demonstrates remarkable effectiveness in:
- Identifying semantic parallels between etymologically related terms across languages
- Maintaining robust performance despite variations in textual context and genre
- Providing quantitative insights into classical philosophical terminology

## Key Results

Our latest analysis shows:
- **81.4% similarity scores** for etymologically related pairs (±0.3%)
- **78.0% baseline similarity** for control pairs (±2.3%)
- **Statistically significant differentiation** (p = 0.012, t = 3.219)
- **Strong performance** across major philosophical concepts:
  - ἐπιστήμη-scientia: 0.820
  - δικαιοσύνη-iustitia: 0.814
  - ἀλήθεια-veritas: 0.814

## Features

**Core Components**
- Multilingual transformer model trained on classical texts
- Cross-lingual term alignment system
- Statistical significance testing framework

**Analysis Tools**
- Automated term extraction and alignment
- Cross-lingual similarity metrics
- Configurable context window analysis

**Visualization Tools**
- Term relationship networks
- Semantic space projections
- Comparative analysis plots

## System Architecture

```mermaid
graph TD
    A[Input Terms] --> B[Text Processor]
    B --> C[Embedding Generator]
    B --> D[Context Extractor]
    
    C --> E[Cross-Lingual Aligner]
    D --> E
    
    E --> F[Similarity Analyzer]
    F --> G[Statistical Validator]
    
    G --> H[Results & Visualizations]
    
    subgraph Data Sources
        I[Greek Texts] --> B
        J[Latin Texts] --> B
    end
    
    subgraph Analysis Pipeline
        B
        C
        D
        E
        F
        G
    end
```

## Installation

```bash
git clone https://github.com/rumiallbert/philoberta.git
cd philoberta
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers
- CLTK (Classical Language Toolkit)
- scikit-learn
- pandas
- matplotlib
- seaborn

## Usage

```python
from philoberta import PhiloBERTa

# Initialize model
model = PhiloBERTa.from_pretrained("agentlab/philoberta-base")

# Analyze terms
results = model.analyze_pair(
    greek_term="ψυχή",
    latin_term="anima",
    context_window=5
)

# Generate visualizations
model.plot_similarity_distribution(results)
```

## Citation

```bibtex
@inproceedings{philoberta2025,
    title = "PhiloBERTA: A Transformer-Based Cross-Lingual Analysis of Greek and Latin Lexicon",
    author = "Allbert, Rumi A.",
    month = feb,
    year = "2025"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.