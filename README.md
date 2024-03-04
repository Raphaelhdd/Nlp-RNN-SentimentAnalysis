# Nlp-RNN-SentimentAnalysis

This repository contains implementations of different models for sentiment analysis, along with training and 
evaluation scripts. 
The models implemented include:

1. **Log-Linear Model with One-Hot Representation**: This model represents words using one-hot encoding and employs a 
simple log-linear architecture for sentiment analysis.

2. **Log-Linear Model with Word Embeddings (Word2Vec)**: This model utilizes pre-trained Word2Vec word embeddings 
to represent words and then applies a log-linear architecture for sentiment analysis.

3. **LSTM Model with Word Embeddings (Word2Vec)**: This model employs a Long Short-Term Memory (LSTM) neural network 
architecture with pre-trained Word2Vec word embeddings for sentiment analysis.

## Setup

To run the code, you need Python 3.x along with the following dependencies:

- `torch`: PyTorch library for deep learning
- `numpy`: Numerical computing library
- `matplotlib`: Plotting library for visualization

You can install the dependencies using pip:

```bash
pip install torch numpy matplotlib
```

## Usage

To train and evaluate the models, you can execute the following scripts:

- `train_log_linear_with_one_hot.py`: Train and evaluate the Log-Linear model with one-hot representation.
- `train_log_linear_with_w2v.py`: Train and evaluate the Log-Linear model with Word2Vec word embeddings.
- `train_lstm_with_w2v.py`: Train and evaluate the LSTM model with Word2Vec word embeddings.

You can run each script using Python:

```bash
python train_log_linear_with_one_hot.py
python train_log_linear_with_w2v.py
python train_lstm_with_w2v.py
```

## Results

The training and evaluation results, including loss and accuracy curves, will be displayed during execution. Additionally, the models' performance on subsets of the test dataset (e.g., negated polarity examples, rare words examples) will be reported.

This exercise is part of the curriculum for the Natural Language Processing course (NLP 67658) at the Hebrew University of Jerusalem in 2024.
