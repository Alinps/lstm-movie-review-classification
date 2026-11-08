from pathlib import Path

from src.preprocessing import load_and_prepare_data
from src.model import build_model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR/"models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Step 1: Load and preprocess data

print("Loading dataset...")

(
    X_train,
    X_val,
    y_train,
    y_val,
    X_test,
    y_test
) =  load_and_prepare_data()

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Testing shape: {X_test.shape}")

# Step 2: Build the model

print("Building LSTM model...")

model = build_model()

model.summary()

# Step 3: Configure callbacks

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=0.00001
)


# Step 4: Train model

print("Starting model training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[early_stopping,reduce_lr]
)

# Step 5: Save trained model

model_path = MODEL_DIR/"imdb_sentiment_lstm.keras"
model.save(model_path)

print(f"Model saved successfully: {model_path}")