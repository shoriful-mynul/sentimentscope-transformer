# SentimentScope: Sentiment Analysis Using Transformers

A transformer-based sentiment classification project built with PyTorch and the IMDB movie review dataset.

## Overview

SentimentScope is a binary sentiment analysis system that classifies movie reviews as either **positive** or **negative**.

The project was developed in the context of CineScope, an entertainment company looking to better understand user sentiment and use those insights to support more personalized movie and content recommendations.

Rather than using a pre-built classification model, this project adapts a custom transformer architecture for sequence-level sentiment classification.

## Project Objectives

The project focuses on:

* Loading and exploring the IMDB movie review dataset
* Preparing labeled text data for transformer-based classification
* Tokenizing reviews using the `bert-base-uncased` tokenizer
* Building custom PyTorch `Dataset` and `DataLoader` implementations
* Adapting a transformer architecture for binary classification
* Training the model using cross-entropy loss and AdamW
* Monitoring validation performance during training
* Evaluating the final model on an unseen test set

## Dataset

The project uses the **IMDB Large Movie Review Dataset**, containing 50,000 labeled movie reviews:

* 25,000 training reviews
* 25,000 test reviews
* Positive reviews are labeled `1`
* Negative reviews are labeled `0`

The training set is further divided into training and validation subsets.

Dataset source:

https://ai.stanford.edu/~amaas/data/sentiment/

## Model Architecture

The project uses a custom transformer model implemented in PyTorch.

The architecture is adapted from a generation-style transformer to perform binary sequence classification.

The main modifications include:

1. **BERT-based subword tokenization**

   Reviews are tokenized using the `bert-base-uncased` tokenizer with padding and truncation.

2. **Transformer representation**

   Token embeddings are processed through the transformer architecture.

3. **Mean pooling**

   The token-level representations are aggregated into a single representation for the complete review.

4. **Classification head**

   The pooled representation is passed through a linear layer to produce two logits corresponding to the negative and positive classes.

## Training

The model was trained using:

* **Framework:** PyTorch
* **Tokenizer:** `bert-base-uncased`
* **Optimizer:** AdamW
* **Learning rate:** `3e-4`
* **Epochs:** 5
* **Loss function:** Cross-Entropy Loss
* **Task:** Binary sentiment classification

Validation accuracy was monitored after every epoch.

### Validation Performance

| Epoch | Validation Accuracy |
| ----: | ------------------: |
|     1 |              72.80% |
|     2 |              76.88% |
|     3 |              78.60% |
|     4 |              77.12% |
|     5 |              78.92% |

The best validation accuracy observed during training was **78.92%** at epoch 5.

## Test Results

After training, the final model was evaluated on the held-out IMDB test dataset.

**Test Accuracy: 77.35%**

The project requirement was to achieve an accuracy greater than 75%, which the final model achieved.

## Repository Structure

```text
sentimentscope-transformer/
│
├── SentimentScope.ipynb
├── sentimentscope_model.pth
└── README.md
```

### Files

**`SentimentScope.ipynb`**

Contains the complete project workflow, including:

* Dataset loading and exploration
* Data preprocessing
* Tokenization
* PyTorch Dataset and DataLoader implementation
* Transformer architecture
* Accuracy calculation
* Training loop
* Validation
* Final test evaluation
* Project conclusion

**`sentimentscope_model.pth`**

Contains the trained model parameters from the model evaluated on the IMDB test set.

## How to Reproduce

1. Clone the repository.

```bash
git clone https://github.com/shoriful-mynul/sentimentscope-transformer.git
cd sentimentscope-transformer
```

2. Open `SentimentScope.ipynb` in Google Colab or a compatible Jupyter environment.

3. Install the required Python packages if they are not already available.

4. Download and extract the IMDB dataset.

5. Run the notebook cells in order.

> Training the transformer from scratch can take significant time depending on the available hardware.

## Technologies Used

* Python
* PyTorch
* Transformers
* Hugging Face Tokenizers
* Pandas
* NumPy
* Matplotlib
* scikit-learn
* Google Colab

## Key Learnings

This project demonstrates how a transformer originally designed around sequence modeling can be adapted for a classification task.

The main conceptual differences from text generation include:

* Predicting a class instead of the next token
* Representing an entire review with a pooled representation
* Using a classification head instead of a vocabulary prediction head
* Training with labeled examples and classification loss
* Evaluating using classification accuracy

The project also provided practical experience with transformer architecture, text tokenization, PyTorch data pipelines, model training, validation, and evaluation.

## Result

**Final Test Accuracy: 77.35%**

The trained transformer successfully exceeded the project's required test accuracy threshold of 75%.

## Author

**Shoriful Islam**

GitHub: https://github.com/shoriful-mynul
