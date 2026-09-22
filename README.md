# SentimentScope: Sentiment Analysis Using Transformers

A transformer-based binary sentiment classification project built with PyTorch and the IMDB Large Movie Review Dataset.

## Project Overview

SentimentScope is a sentiment analysis system designed to classify movie reviews as either **positive** or **negative**.

The project was developed in the context of CineScope, a fictional entertainment company interested in understanding user sentiment from movie reviews and using these insights to support personalized content recommendations.

The project adapts a custom transformer architecture for sequence-level text classification and evaluates its performance on the IMDB test dataset.

## Objectives

The main objectives of this project are to:

* Explore and preprocess the IMDB movie review dataset
* Tokenize text using the `bert-base-uncased` tokenizer
* Build custom PyTorch `Dataset` and `DataLoader` pipelines
* Adapt a transformer architecture for binary sentiment classification
* Train the model using supervised learning
* Monitor validation performance during training
* Evaluate the trained model on an unseen test dataset
* Save the trained model as a reusable checkpoint

## Dataset

The project uses the **IMDB Large Movie Review Dataset**, which contains 50,000 labeled movie reviews:

* **25,000** training reviews
* **25,000** test reviews
* **Positive:** label `1`
* **Negative:** label `0`

The original training set is further divided into training and validation subsets during the project workflow.

### Dataset Source

https://ai.stanford.edu/~amaas/data/sentiment/

## Model Architecture

The project uses a custom transformer model implemented with PyTorch.

The architecture was adapted from a generation-oriented transformer to perform sequence-level binary classification.

### Classification Pipeline

**1. Tokenization**

Movie reviews are tokenized using the `bert-base-uncased` tokenizer. Padding and truncation are applied to prepare the text for transformer processing.

**2. Transformer Representation**

The tokenized inputs are passed through the transformer architecture to generate contextual representations for the input sequence.

**3. Mean Pooling**

The token-level representations are aggregated using mean pooling to obtain a single representation for the complete review.

**4. Classification Head**

The pooled representation is passed through a linear classification layer that produces two logits:

* `0` → Negative
* `1` → Positive

## Training Configuration

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Framework     | PyTorch                         |
| Tokenizer     | `bert-base-uncased`             |
| Optimizer     | AdamW                           |
| Learning Rate | `3e-4`                          |
| Epochs        | 5                               |
| Loss Function | Cross-Entropy Loss              |
| Task          | Binary Sentiment Classification |

## Validation Results

Validation accuracy was evaluated after each training epoch.

| Epoch | Validation Accuracy |
| ----: | ------------------: |
|     1 |              72.80% |
|     2 |              76.88% |
|     3 |              78.60% |
|     4 |              77.12% |
|     5 |              78.92% |

The highest recorded validation accuracy was **78.92%** at epoch 5.

## Test Results

After training, the final model was evaluated on the held-out IMDB test dataset.

### Final Test Accuracy

**77.35%**

The project requirement was to achieve a test accuracy greater than **75%**. The trained model achieved **77.35%** on the test dataset.

## Model Checkpoint

The trained model parameters are provided as a PyTorch checkpoint:

```text
sentimentscope_model.pth
```

The checkpoint is approximately **35 MB** and is provided through the project's **GitHub Release** because the standard GitHub repository web uploader has a file-size limitation.

The checkpoint can be downloaded from the **Releases** section of this repository.

To load the checkpoint, the model architecture must first be defined and initialized in the same way as in the notebook. The saved parameters can then be loaded using:

```python
model.load_state_dict(torch.load("sentimentscope_model.pth", map_location=device))
model.eval()
```

## Repository Structure

```text
sentimentscope-transformer/
├── README.md
└── SentimentScope.ipynb

GitHub Release:
└── v1.0.0
    └── sentimentscope_model.pth
```

> The model checkpoint is distributed as a GitHub Release asset rather than as a regular repository file.

### Files

#### `SentimentScope.ipynb`

The completed Jupyter notebook containing the full project workflow, including:

* Dataset download and loading
* Dataset exploration
* Text preprocessing
* Tokenization
* PyTorch Dataset and DataLoader implementation
* Transformer architecture
* Model initialization
* Training loop
* Validation
* Test evaluation
* Model checkpoint saving
* Final project conclusion

#### `sentimentscope_model.pth`

The trained PyTorch model checkpoint containing the learned model parameters.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/shoriful-mynul/sentimentscope-transformer.git
cd sentimentscope-transformer
```

### 2. Open the notebook

Open:

```text
SentimentScope.ipynb
```

in Google Colab or a compatible Jupyter environment.

### 3. Install the required dependencies

The notebook contains the required package setup and imports.

### 4. Download the dataset

Run the dataset preparation cells in the notebook to download and extract the IMDB dataset.

### 5. Run the notebook

Execute the cells in order to reproduce the data preparation, model construction, training, validation, and test evaluation workflow.

> Training the transformer model can take significant time depending on the available hardware.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Pandas
* NumPy
* Matplotlib
* scikit-learn
* Google Colab
* Jupyter Notebook

## Key Learnings

This project provided practical experience with:

* Transformer-based NLP
* Text tokenization
* Sequence classification
* PyTorch Dataset and DataLoader design
* Transformer architecture adaptation
* Mean pooling for sequence representation
* Classification heads
* Model training and validation
* Test-set evaluation
* Saving and loading PyTorch model checkpoints

A key aspect of the project was adapting a transformer architecture originally intended for sequence modeling to perform binary sentiment classification.

## Result

**Final Test Accuracy: 77.35%**

The trained transformer successfully exceeded the project's required test accuracy threshold of 75%.

## Conclusion

SentimentScope successfully demonstrates a transformer-based approach to binary sentiment classification on the IMDB movie review dataset.

## Author

**Shoriful Islam**

GitHub: https://github.com/shoriful-mynul
