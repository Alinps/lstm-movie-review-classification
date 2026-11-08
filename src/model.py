from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    SpatialDropout1D
)
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Adam

from src.preprocessing import VOCAB_SIZE


def build_model():

    model = Sequential([

        # Embedding Layer
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=16,
            mask_zero=True
        ),

        SpatialDropout1D(0.5),

        # LSTM Layer
        Bidirectional(LSTM(16)),

        Dropout(0.5),

        # Hidden Layer
        Dense(
            units= 16,
            activation="relu",
            kernel_regularizer=l2(1e-4)
        ),

        # Output Layer
        Dense(
            units=1, 
            activation="sigmoid"
        ),

    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model