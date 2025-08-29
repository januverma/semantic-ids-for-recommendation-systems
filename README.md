# Semantic IDs for Recommendation Systems

A PyTorch implementation comparing traditional ID-based and semantic-based recommendation models on the Amazon Beauty dataset. This project explores how semantic representations of items can improve recommendation performance, particularly for cold-start scenarios.

## Overview

This repository implements and compares multiple recommendation approaches:

1. **Matrix Factorization (MF) Models**: Traditional collaborative filtering with user/item embeddings vs semantic item representations
2. **Neural CTR Models**: Deep learning models (WDL, DeepFM, DLRM) comparing baseline item features vs semantic encodings

## Project Structure

```
├── src/                          # Core model implementations
│   ├── mf_models.py             # Matrix factorization models (ID vs Semantic)
│   ├── neural_models.py         # Neural CTR models (WDL, DeepFM, DLRM)
│   └── tiger_model.py           # TIGER model implementation
├── utils/                        # Utility functions
│   ├── common_utils.py          # General utilities (seed setting, data splitting)
│   ├── mf_data_utils.py         # Data utilities for MF models
│   ├── mf_utils.py              # MF-specific utilities and metrics
│   ├── neural_data_utils.py     # Data utilities for neural models
│   └── neural_utils.py          # Neural model utilities and metrics
├── beauty/                       # Amazon Beauty dataset (excluded from git)
│   ├── meta.json.gz             # Item metadata
│   ├── rating_splits_augmented.pkl  # Train/val/test splits
│   ├── datamaps.json            # User/item ID mappings
│   ├── semantic_ids_fixed.json  # Generated semantic ID mappings
│   ├── semantic_codebooks.pkl   # Trained quantization codebooks
│   └── sequential_data.txt      # Sequential interaction data
├── semantic_id_generator.py     # Main semantic ID generation module and script
├── analyze_semantic_ids.py      # Analysis and visualization of semantic IDs
├── mf_train_eval.py             # Training script for MF models
├── neural_train_eval.py         # Training script for neural models
├── tiger_train_eval.py          # Training script for TIGER model
└── results/                     # Output directory for results
```

## Models

### Matrix Factorization Models

- **MF_ID_Torch**: Traditional biased matrix factorization with user/item embeddings
- **MF_Semantic_Torch**: Uses compositional item representations from semantic codes with learned level weights

### Neural CTR Models

- **WDL (Wide & Deep Learning)**: Combines wide linear model with deep neural networks
- **DeepFM**: Deep factorization machine with first-order, second-order, and deep components
- **DLRM (Deep Learning Recommendation Model)**: Facebook's recommendation model with categorical feature interactions

Each neural model supports two modes:
- **Baseline**: Uses traditional item IDs and metadata features
- **Semantic**: Replaces item IDs with multi-level semantic codes

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd semantic-ids-for-recommendation-systems

# Install dependencies
pip install torch pandas numpy scikit-learn sentence-transformers
```

## Usage

### 1. Generate Semantic IDs (Required First Step)

Before running the recommendation models, you need to generate semantic IDs from the product metadata:

```bash
python semantic_id_generator.py
```

This will:
- Load Amazon Beauty product metadata (`beauty/meta.json.gz`)
- Extract text features (title, brand, categories, description)
- Generate sentence embeddings using SentenceTransformers
- Apply KMeans residual quantization to create semantic IDs
- Handle collisions with suffix tokens
- Save semantic IDs to `beauty/semantic_ids.json`
- Save trained codebooks to `beauty/semantic_codebooks.pkl`

**Output**: 4-level semantic IDs (3 quantization levels + 1 collision suffix)

### 2. Analyze Semantic IDs (Optional)

Visualize and analyze the generated semantic IDs:

```bash
python analyze_semantic_ids.py
```

This will:
- Analyze semantic ID distribution across levels
- Detect and visualize collision patterns
- Calculate semantic similarity metrics
- Generate plots in the `plots/` directory
- Show usage examples for model integration

### 3. Train Recommendation Models

#### Matrix Factorization Models

```bash
python mf_train_eval.py
```

This will:
- Load the Amazon Beauty dataset and semantic IDs
- Train both ID-based and semantic MF models
- Evaluate on test set with overall and slice-based metrics
- Save results to `results/mf_results.json`

#### Neural CTR Models

```bash
python neural_train_eval.py
```

This will:
- Train all three neural models (WDL, DeepFM, DLRM) in both baseline and semantic modes
- Evaluate performance including cold-start scenarios
- Save results with timestamp

#### TIGER Model

```bash
python tiger_train_eval.py
```

## Data Format

The project expects the Amazon Beauty dataset in the following format:
- `meta.json.gz`: Item metadata including categories, brand, description
- `rating_splits_augmented.pkl`: Pre-split rating data (train/val/test)
- `datamaps.json`: User/item ID mappings
- `semantic_ids_fixed.json`: Multi-level semantic codes for items

## Key Features

- **Semantic ID Generation**: KMeans residual quantization of item metadata embeddings
- **Collision Handling**: Automatic resolution of semantic ID collisions with suffix tokens
- **Multi-level Encoding**: 4-level semantic IDs (3 quantization + 1 collision resolution)
- **Temporal Splitting**: Uses temporal split per user for realistic evaluation
- **Cold-start Analysis**: Evaluates performance on items with limited training data
- **Comprehensive Metrics**: Reports RMSE, AUC, F1, and slice-based performance
- **GPU Support**: CUDA and MPS (Apple Silicon) acceleration
- **Visualization Tools**: Analysis and plotting of semantic ID distributions

## Evaluation Metrics

- **Overall Performance**: RMSE (regression), AUC/F1 (classification)
- **Cold-start Performance**: Metrics on items with few training interactions
- **Slice Analysis**: Performance across different user/item popularity segments

## Results

Results are saved in JSON format containing:
- Overall test performance
- Cold-start item performance  
- Slice-based analysis by popularity
- Training convergence metrics

## Semantic ID Generation Process

The semantic ID generation follows this pipeline:

1. **Text Extraction**: Combines item title, brand, categories, and description into unified text
2. **Embedding Generation**: Uses SentenceTransformers to create dense vector representations
3. **Residual Quantization**: Applies KMeans clustering in multiple levels to create discrete codes
4. **Collision Resolution**: Appends suffix tokens to handle items with identical semantic IDs
5. **Storage**: Saves both the semantic IDs and trained codebooks for reproducibility

The resulting 4-level semantic IDs can be used as compositional item representations in recommendation models.

## Contributing

This is a research project exploring semantic representations in recommendation systems. The code is designed for experimental comparison of different approaches on the Amazon Beauty dataset.