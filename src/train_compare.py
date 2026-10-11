import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# 1. Load dataset
# -----------------------------

DATA_PATH = "data/processed/news_dataset.csv"

df = pd.read_csv(DATA_PATH)

X = df["combined_text"].fillna("")
y = df["label"]

# -----------------------------
# 2. Split dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# 3. TF-IDF transformation
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

print("TF-IDF transformation completed")
print("Feature count:", X_train_tfidf.shape[1])

# -----------------------------
# 4. Define models
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Linear SVM": LinearSVC(
        random_state=42
    ),
    "Multinomial Naive Bayes": MultinomialNB()
}

results = []

os.makedirs("models", exist_ok=True)

# -----------------------------
# 5. Train and evaluate models
# -----------------------------

for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    start_time = time.time()

    model.fit(X_train_tfidf, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Fake News", "Real News"]
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Training Time (seconds)": training_time
    })

    filename = model_name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(model, f"models/{filename}")

# -----------------------------
# 6. Save vectorizer
# -----------------------------

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer_compare.pkl"
)

# -----------------------------
# 7. Display comparison
# -----------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-Score": "{:.4f}".format,
            "Training Time (seconds)": "{:.2f}".format
        }
    )
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("results/model_comparison.csv")