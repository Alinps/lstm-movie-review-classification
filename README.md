# Movie Review Sentiment Analysis using LSTM

A Deep Learning project that performs binary sentiment classification on movie reviews using TensorFlow and Keras. The model uses an Embedding Layer to learn dense word representations and an LSTM (Long Short-Term Memory) network to understand sequential patterns in text.

The model classifies movie reviews into two sentiment categories:

- **Positive (1)**
- **Negative (0)**

---

## Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) task that identifies the emotional tone or opinion expressed in textual data.

In this project, an LSTM-based neural network is developed to classify movie reviews based on their sentiment.

The project uses the IMDB Movie Reviews dataset, which contains 50,000 movie reviews divided into training and testing datasets.

The complete workflow includes:

1. Dataset loading and exploration.
2. Text preprocessing and integer encoding.
3. Sequence padding and truncation.
4. Word embedding using TensorFlow's Embedding Layer.
5. Sequential learning using LSTM.
6. Model training and hyperparameter configuration.
7. Model evaluation using classification metrics.
8. Sentiment prediction on new movie reviews.

---

## Model Architecture

The model uses the following architecture:

```text
             Movie Review
                   |
                   v
          Text Preprocessing
                   |
                   v
           Integer Encoding
                   |
                   v
          Padding Sequences
                   |
                   v
             Embedding
                   |
                   v
                LSTM
                   |
                   v
              Dense Layer
                   |
                   v
          Sigmoid Activation
                   |
                   v
          Sentiment Prediction
                   |
             -------------
             |           |
          Positive     Negative
```

### Architecture Components

| Layer | Purpose |
|---|---|
| Embedding Layer | Converts integer word IDs into dense numerical vectors. |
| LSTM Layer | Learns sequential patterns and contextual relationships in movie reviews. |
| Dense Layer | Performs binary sentiment classification. |
| Sigmoid Activation | Produces a probability between 0 and 1. |

---

## Dataset

**Dataset:** IMDB Movie Reviews

The dataset is available through TensorFlow Keras.

It contains:

| Dataset | Number of Reviews |
|---|---:|
| Training | 25,000 |
| Testing | 25,000 |
| Total | 50,000 |

### Sentiment Labels

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Positive |

The dataset is already tokenized and integer encoded.

The project uses the 10,000 most frequent vocabulary entries.

Dataset loading:

```python
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)
```

The dataset is automatically downloaded and cached by Keras when required.

---

## Project Structure

```text
movie-review-sentiment-analysis/
│
├── notebooks/
│   ├── lstm_sentiment_analysis_imdb_movie_review.ipynb
│   
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── imdb_sentiment_lstm.keras
│
├── results/
│   └── confusion_matrix.png
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```
