from pathlib import Path

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from src.preprocessing import load_and_prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models"/"imdb_sentiment_lstm.keras"
RESULT_DIR = BASE_DIR/"results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# Step 1: Load test data

print("Loading test data...")

(
    X_train,
    X_val,
    y_train,
    y_val,
    X_test,
    y_test
) = load_and_prepare_data()

print("Loading trained model...")

model = load_model(MODEL_PATH)

# Step 3: Evaluate model

test_loss,test_accuracy = model.evaluate(X_test, y_test) 


print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Step 4: Generate predictions

y_probability = model.predict(X_test)

y_pred = (y_probability >= 0.5).astype(int).ravel()

# Step 5: Calculate matrics

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")
print(classification_report(y_test,y_pred,target_names=["Negative", "Positive"]))

# Step 6: Confusion matix

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative","Positive"]
    )

display.plot()

plt.title("IMDB Sentiment Analysis Confusion Matrix")

plt.savefig(
    RESULT_DIR/"confusion_matrix.png",
    bbox_inches="tight"
)

plt.show()