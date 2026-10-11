import joblib
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "linear_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer_compare.pkl"


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

print("Loading trained model and TF-IDF vectorizer...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Model and vectorizer loaded successfully.")


# ============================================================
# GET FEATURE NAMES AND MODEL COEFFICIENTS
# ============================================================

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]


# ============================================================
# MOST IMPORTANT FEATURES
# ============================================================

# Features with the lowest coefficients are associated with Fake News
fake_indices = np.argsort(coefficients)[:30]

# Features with the highest coefficients are associated with Real News
real_indices = np.argsort(coefficients)[-30:][::-1]


# ============================================================
# DISPLAY FAKE NEWS FEATURES
# ============================================================

print("\n" + "=" * 60)
print("WORDS MOST ASSOCIATED WITH FAKE NEWS")
print("=" * 60)

for index in fake_indices:
    print(
        f"{feature_names[index]:30s} "
        f"{coefficients[index]:.6f}"
    )


# ============================================================
# DISPLAY REAL NEWS FEATURES
# ============================================================

print("\n" + "=" * 60)
print("WORDS MOST ASSOCIATED WITH REAL NEWS")
print("=" * 60)

for index in real_indices:
    print(
        f"{feature_names[index]:30s} "
        f"{coefficients[index]:.6f}"
    )


# ============================================================
# EXPLAIN A SAMPLE INPUT
# ============================================================

sample_title = "Government announces new education policy"

sample_text = (
    "The government announced a new education policy aimed at improving "
    "school infrastructure, teacher training, and student learning outcomes. "
    "Officials stated that the policy would be implemented in phases over "
    "the next three years."
)

sample_combined_text = sample_title + " " + sample_text

sample_vector = vectorizer.transform([sample_combined_text])

prediction = model.predict(sample_vector)[0]
decision_score = model.decision_function(sample_vector)[0]

print("\n" + "=" * 60)
print("SAMPLE PREDICTION EXPLANATION")
print("=" * 60)

print(
    "Prediction:",
    "Fake News" if prediction == 0 else "Real News"
)

print(f"Decision score: {decision_score:.4f}")


# ============================================================
# FIND CONTRIBUTING WORDS IN THE INPUT
# ============================================================

nonzero_indices = sample_vector.nonzero()[1]

contributions = []

for index in nonzero_indices:

    tfidf_value = sample_vector[0, index]
    contribution = tfidf_value * coefficients[index]

    contributions.append(
        (
            feature_names[index],
            float(tfidf_value),
            float(contribution)
        )
    )


# Sort by contribution
contributions.sort(key=lambda item: item[2])


print("\nWords contributing toward Fake News:")

fake_contributions = [
    item for item in contributions if item[2] < 0
]

for word, tfidf_value, contribution in fake_contributions[:10]:
    print(
        f"{word:25s} "
        f"Contribution: {contribution:.6f}"
    )


print("\nWords contributing toward Real News:")

real_contributions = [
    item for item in contributions if item[2] > 0
]

for word, tfidf_value, contribution in reversed(real_contributions[-10:]):
    print(
        f"{word:25s} "
        f"Contribution: {contribution:.6f}"
    )