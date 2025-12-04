# PLDAC-RECO: Board Game Recommendation System

A comprehensive recommendation system for board games using collaborative filtering, matrix factorization, and deep learning techniques. This project analyzes board game ratings and reviews from Tric Trac to provide personalized game recommendations and generate automated review text.

## 📋 Project Overview

This project implements multiple approaches to board game recommendation:

- **Collaborative Filtering**: User-based and item-based recommendation using KNN and Surprise library
- **Matrix Factorization**: SVD, NMF, and Baseline models for rating prediction
- **Deep Learning**: LSTM-based text generation for creating game reviews
- **Exploratory Data Analysis**: Comprehensive analysis of user preferences, game popularity, and rating patterns

## 🚀 Features

- **Rating Prediction**: Predict user ratings for board games they haven't played
- **Recommendation Engine**: Generate personalized game recommendations based on user preferences
- **Review Generation**: AI-generated game reviews using character-level RNN
- **Data Analysis**: In-depth exploratory data analysis with visualizations
- **Multiple Algorithms**: Compare performance across various recommendation algorithms

## 📊 Dataset

The project uses data scraped from Tric Trac containing:
- **Board game details**: Metadata, categories, and game information
- **User reviews**: Ratings and text reviews from users
- **User information**: User profiles and rating patterns

Data is stored in BSON format and includes:
- `details.bson`: Game metadata and statistics
- `avis.bson`: User reviews and ratings
- `jeux.bson`: Game information
- `infos_scrapping.bson`: Additional scraped information

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/BidoElguindy/PLDAC-RECO.git
cd PLDAC-RECO
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for text analysis):
```python
import nltk
nltk.download('stopwords')
```

4. (Optional) For spaCy NLP features:
```bash
python -m spacy download en_core_web_sm
```

## 📁 Project Structure

```
PLDAC-RECO/
├── EDA.ipynb                      # Exploratory Data Analysis
├── ratings_pred.ipynb             # Rating prediction models
├── lstm.ipynb                     # LSTM text generation
├── analyse_commentaire.ipynb      # Comment analysis
├── gen.ipynb                      # General experiments
├── notes.ipynb                    # Project notes and findings
├── fonctions.py                   # Utility functions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 💻 Usage

### 1. Exploratory Data Analysis
Start with `EDA.ipynb` to understand the dataset:
```bash
jupyter notebook EDA.ipynb
```

### 2. Rating Prediction
Use `ratings_pred.ipynb` to train and evaluate recommendation models:
- Implements KNN, SVD, NMF, and Baseline algorithms
- Custom train/test split and evaluation metrics
- Performance comparison across different game popularity levels

### 3. Text Generation
Explore `lstm.ipynb` for AI-generated game reviews:
- Character-level RNN using PyTorch
- Review generation based on rating ranges
- Customizable text generation parameters

### 4. Utility Functions
The `fonctions.py` module provides:
- `minipingpong()`: Data filtering algorithm for sparse matrices
- `subtract_mean()`: User bias normalization
- Evaluation metrics and data preprocessing utilities

## 🔬 Key Algorithms

### Collaborative Filtering
- **KNN-based**: User-based and item-based nearest neighbors
- **SVD**: Singular Value Decomposition for matrix factorization
- **NMF**: Non-negative Matrix Factorization
- **Baseline**: Mean-based predictions with bias terms

### Deep Learning
- **LSTM**: Character-level recurrent neural network
- **Text Generation**: Automated review generation conditioned on ratings

## 📈 Results

The project explores:
- Impact of data sparsity on model performance
- Rating prediction across different game popularity levels
- User and item bias in rating patterns
- Quality of AI-generated reviews

Detailed findings and analysis are documented in `notes.ipynb`.

## 🤝 Contributing

This is an academic project developed for the PLDAC course. Contributions and suggestions are welcome!

