from pathlib import Path

from tensorflow.keras.models import load_model

from src.preprocessing import preprocessing_review


# Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR/"models"/"imdb_sentiment_lstm.keras"

# Step 1: Load trained model

print("Loading sentiment analysis model...")

model = load_model(MODEL_PATH)

# Step 2: Prediction function 

def predict_sentiment(review):

    processed_review = preprocessing_review(review)

    # Generate probability
    probability = model.predict(
        processed_review,
        verbose=0
    )[0][0]

    # Classify sentiment

    if probability >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    print(f"\nMovie Review:{review}")
    print(f"Sentiment: {sentiment}")
    print(f"Positive probability: {round(float(probability), 4)}")

# Step 3: Test predictions

if __name__ =="__main__":
    
    review = input("Enter a movie review: ")

    predict_sentiment(review)