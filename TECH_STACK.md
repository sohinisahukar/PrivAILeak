# 🛠️ Technology Stack & Libraries

Complete list of all libraries, frameworks, and technologies used in the PrivAI-Leak project.

---

## 📋 Table of Contents
1. [Core ML/DL Frameworks](#core-mldl-frameworks)
2. [NLP & Language Models](#nlp--language-models)
3. [Privacy & Security](#privacy--security)
4. [Data Processing](#data-processing)
5. [Visualization](#visualization)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Utilities](#utilities)
8. [Optional/Production Testing](#optionalproduction-testing)
9. [Python Version](#python-version)

---

## 🤖 Core ML/DL Frameworks

### PyTorch (`torch>=2.0.0`)
**Purpose:** Deep learning framework
- **Used for:**
  - Model training (baseline and DP models)
  - Gradient computation and optimization
  - Tensor operations
  - GPU/CPU device management
- **Key Features:**
  - Automatic differentiation
  - Neural network layers
  - Optimizers (AdamW)
  - DataLoader for batching

### NumPy (`numpy>=1.24.0`)
**Purpose:** Numerical computing
- **Used for:**
  - Mathematical operations
  - Array manipulations
  - Privacy accounting calculations
  - Statistical computations

---

## 🗣️ NLP & Language Models

### Hugging Face Transformers (`transformers>=4.30.0`)
**Purpose:** Pre-trained language models and tokenizers
- **Used for:**
  - GPT-2 model loading (`GPT2LMHeadModel`)
  - Tokenization (`GPT2Tokenizer`)
  - Text generation
  - Model fine-tuning
- **Models Used:**
  - `gpt2` (124M parameters) - Baseline model
  - `distilgpt2` (82M parameters) - Alternative/DP model option

### Hugging Face Datasets (`datasets>=2.14.0`)
**Purpose:** Dataset management
- **Used for:**
  - Dataset loading utilities
  - Data preprocessing helpers

---

## 🔒 Privacy & Security

### Opacus (`opacus>=1.4.0`)
**Purpose:** Differential Privacy for PyTorch
- **Used for:**
  - Privacy Engine (`PrivacyEngine`)
  - RDP Accountant (`RDPAccountant`)
  - Batch memory management
  - **Note:** Currently using manual DP-SGD implementation, but Opacus available for reference

### Custom RDP Accountant
**Purpose:** Renyi Differential Privacy accounting
- **Implementation:** Custom `RDPAccountant` class in `dp_training_manual.py`
- **Used for:**
  - Tracking privacy budget consumption
  - Converting RDP to (ε, δ)-DP bounds
  - Privacy composition tracking

---

## 📊 Data Processing

### Pandas (`pandas>=2.0.0`)
**Purpose:** Data manipulation and analysis
- **Used for:**
  - Creating comparison tables
  - DataFrames for results
  - CSV export/import
  - Data aggregation

### Faker (`faker>=18.0.0`)
**Purpose:** Synthetic data generation
- **Used for:**
  - Generating fake patient names
  - Creating synthetic emails, SSNs, phone numbers
  - Generating addresses and dates
  - Healthcare-specific data (MRNs, conditions, medications)

### Scikit-learn (`scikit-learn>=1.3.0`)
**Purpose:** Machine learning utilities
- **Used for:**
  - Statistical analysis
  - Evaluation metrics
  - Data preprocessing helpers

---

## 📈 Visualization

### Matplotlib (`matplotlib>=3.7.0`)
**Purpose:** Plotting and visualization
- **Used for:**
  - Privacy-utility trade-off plots
  - Comparison charts
  - Bar graphs
  - Line plots
- **Output:** PNG files (300 DPI)

### Seaborn (`seaborn>=0.12.0`)
**Purpose:** Statistical visualization
- **Used for:**
  - Enhanced plot styling
  - Statistical plots
  - Better color schemes
  - Grid layouts

### Plotly (`plotly>=5.14.0`)
**Purpose:** Interactive visualizations
- **Status:** Listed in requirements but not actively used
- **Potential use:** Interactive dashboards

---

## 📏 Evaluation Metrics

### ROUGE Score (`rouge-score>=0.1.2`)
**Purpose:** Text generation quality metrics
- **Used for:**
  - Evaluating generated text quality
  - Comparing model outputs

### NLTK (`nltk>=3.8.0`)
**Purpose:** Natural Language Toolkit
- **Used for:**
  - Text preprocessing
  - Tokenization
  - Language processing utilities

### SacreBLEU (`sacrebleu>=2.3.1`)
**Purpose:** BLEU score calculation
- **Used for:**
  - Text quality evaluation
  - Standardized BLEU metrics

---

## 🛠️ Utilities

### tqdm (`tqdm>=4.65.0`)
**Purpose:** Progress bars
- **Used for:**
  - Training progress indicators
  - Data loading progress
  - Epoch progress bars
  - Attack simulation progress

### python-dotenv (`python-dotenv>=1.0.0`)
**Purpose:** Environment variable management
- **Used for:**
  - API key management
  - Configuration via `.env` files
  - Secure credential handling

### Standard Library Modules
- **`json`**: JSON file I/O
- **`pathlib`**: File path handling
- **`argparse`**: Command-line argument parsing
- **`typing`**: Type hints
- **`random`**: Random number generation
- **`sys`**: System-specific parameters
- **`collections`**: Specialized data structures

---

## 🌐 Optional/Production Testing

### OpenAI (`openai>=1.0.0`)
**Purpose:** GPT-4 API access
- **Status:** Optional
- **Used for:** Testing production models (GPT-4, GPT-3.5)

### Anthropic (`anthropic>=0.18.0`)
**Purpose:** Claude API access
- **Status:** Optional
- **Used for:** Testing Claude models

### Google Generative AI (`google-generativeai>=0.3.0`)
**Purpose:** Gemini API access
- **Status:** Optional
- **Used for:** Testing Google's Gemini models

---

## 🐍 Python Version

- **Python:** 3.8+ (as specified in README)
- **Recommended:** Python 3.9 or higher

---

## 📦 Installation

All dependencies can be installed via:
```bash
pip install -r requirements.txt
```

---

## 🏗️ Architecture Overview

### Technology Stack Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   PrivAI-Leak Pipeline                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Data Layer:                                             │
│  ├── Faker (synthetic data generation)                 │
│  ├── Pandas (data manipulation)                         │
│  └── NumPy (numerical operations)                       │
│                                                          │
│  ML Layer:                                               │
│  ├── PyTorch (deep learning framework)                  │
│  ├── Transformers (GPT-2 models)                        │
│  └── Custom DP-SGD (privacy-preserving training)        │
│                                                          │
│  Privacy Layer:                                          │
│  ├── Custom RDP Accountant (privacy tracking)          │
│  ├── Per-sample gradient clipping                       │
│  └── Gaussian noise addition                            │
│                                                          │
│  Evaluation Layer:                                        │
│  ├── Privacy attacks (membership inference)            │
│  ├── PII detection                                       │
│  └── Perplexity calculation                             │
│                                                          │
│  Visualization Layer:                                     │
│  ├── Matplotlib (static plots)                          │
│  └── Seaborn (statistical plots)                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Technologies by Component

### 1. **Data Generation**
- **Faker**: Synthetic patient data
- **Pandas**: Data structuring
- **Random**: Shuffling and sampling

### 2. **Model Training**
- **PyTorch**: Core training loop
- **Transformers**: GPT-2 models
- **Custom DP-SGD**: Privacy-preserving training

### 3. **Privacy Implementation**
- **Custom RDP Accountant**: Privacy accounting
- **PyTorch**: Per-sample gradients
- **NumPy**: Privacy calculations

### 4. **Privacy Attacks**
- **Transformers**: Model inference
- **PyTorch**: Gradient-based attacks
- **Custom PII detection**: Pattern matching

### 5. **Evaluation**
- **Pandas**: Results tables
- **NumPy**: Statistical analysis
- **Custom metrics**: Leakage rates, privacy risk

### 6. **Visualization**
- **Matplotlib**: Plot generation
- **Seaborn**: Styling
- **Pandas**: Data preparation

---

## 📝 Version Compatibility

| Library | Minimum Version | Purpose |
|---------|----------------|---------|
| Python | 3.8+ | Language runtime |
| PyTorch | 2.0.0+ | Deep learning |
| Transformers | 4.30.0+ | NLP models |
| NumPy | 1.24.0+ | Numerical ops |
| Pandas | 2.0.0+ | Data processing |

---

## 🎯 Primary Technologies Summary

**Core Stack:**
- 🐍 **Python 3.8+**
- 🔥 **PyTorch 2.0+**
- 🤗 **Hugging Face Transformers**
- 🔒 **Custom DP-SGD Implementation**

**Supporting Libraries:**
- 📊 **Pandas & NumPy** (data)
- 📈 **Matplotlib & Seaborn** (visualization)
- 🎲 **Faker** (synthetic data)
- 📏 **tqdm** (progress bars)

**Privacy Technologies:**
- 🔐 **Renyi Differential Privacy (RDP)**
- 📊 **Custom RDP Accountant**
- ✂️ **Per-sample gradient clipping**
- 🔊 **Gaussian noise mechanism**

---

**Last Updated:** $(date)  
**Status:** Current as of latest implementation

