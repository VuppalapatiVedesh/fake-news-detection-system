import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# 1. Load processed dataset
# -----------------------------

DATA_PATH = "data/processed/news_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# -----------------------------
# 2. Select input and target
# -----------------------------

# Use only title + text.
# Do not use subject because it causes data leakage.
X = df["combined_text"].fillna("")
y = df["label"]

# -----------------------------
# 3. Split the dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# 4. Convert text into TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training matrix shape:", X_train_tfidf.shape)
print("TF-IDF testing matrix shape:", X_test_tfidf.shape)

# -----------------------------
# 5. Train Logistic Regression
# -----------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed successfully")

# -----------------------------
# 6. Make predictions
# -----------------------------

y_pred = model.predict(X_test_tfidf)

# -----------------------------
# 7. Evaluate the model
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake News", "Real News"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 8. Save model and vectorizer
# -----------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel saved to:")
print("models/logistic_regression_model.pkl")

print("\nVectorizer saved to:")
print("models/tfidf_vectorizer.pkl")