import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from sklearn.model_selection import train_test_split


# Configuration
VOCAB_SIZE = 10000
MAX_LENGTH = 250


def load_and_prepare_data():

    # Load imdb dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = VOCAB_SIZE)

    # Apply padding
    X_train = pad_sequences(
        X_train,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    X_test = pad_sequences(
        X_test,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Split training data into training and validation 
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, y_train, y_val, X_test, y_test



def preprocessing_review(review):

    # Load original word index mapping
    word_index = imdb.get_word_index()

    # Tokenize raw text
    words = text_to_word_sequence(review)

    # Convert words into IMDB integer IDs
    encoded_review = [1]

    for word in words:
        
        index = word_index.get(word,2)

        if index + 3 < VOCAB_SIZE:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)
        
    # Apply padding
    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return padded_review