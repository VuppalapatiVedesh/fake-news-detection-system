import matplotlib

# IMPORTANT: Set backend BEFORE importing pyplot
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "news_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. Create text length features
# --------------------------------------------------

df["title_length"] = df["title"].astype(str).str.split().str.len()
df["text_length"] = df["text"].astype(str).str.split().str.len()
df["combined_length"] = df["combined_text"].astype(str).str.split().str.len()

# --------------------------------------------------
# 4. Class distribution
# --------------------------------------------------

print("\n========== CLASS DISTRIBUTION ==========")
print(df["label"].value_counts())

print("\nClass percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

# --------------------------------------------------
# 5. Length statistics
# --------------------------------------------------

print("\n========== TEXT LENGTH STATISTICS ==========")
print(
    df.groupby("label")[
        ["title_length", "text_length", "combined_length"]
    ].describe()
)

# --------------------------------------------------
# 6. Subject distribution
# --------------------------------------------------

print("\n========== SUBJECT DISTRIBUTION ==========")
print(df["subject"].value_counts())

print("\nSubject distribution by label:")
print(pd.crosstab(df["subject"], df["label"]))

# --------------------------------------------------
# 7. Plot class distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

sns.countplot(data=df, x="label", ax=ax)

ax.set_title("Class Distribution")
ax.set_xlabel("Label (0 = Fake, 1 = Real)")
ax.set_ylabel("Number of Articles")

fig.tight_layout()
fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# 8. Plot title length distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    data=df,
    x="title_length",
    hue="label",
    bins=50,
    kde=True,
    ax=ax
)

ax.set_title("Title Length Distribution")
ax.set_xlabel("Number of Words")
ax.set_ylabel("Number of Articles")

fig.tight_layout()
fig.savefig(RESULTS_DIR / "title_length_distribution.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# 9. Plot article length distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    data=df,
    x="text_length",
    hue="label",
    bins=50,
    kde=True,
    ax=ax
)

ax.set_title("Article Length Distribution")
ax.set_xlabel("Number of Words")
ax.set_ylabel("Number of Articles")

fig.tight_layout()
fig.savefig(RESULTS_DIR / "text_length_distribution.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# 10. Most common words
# --------------------------------------------------

def get_common_words(text_series, top_n=20):
    words = []

    for text in text_series:
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        words.extend(text.split())

    return Counter(words).most_common(top_n)


print("\n========== MOST COMMON WORDS ==========")

print("\nFake news:")
print(get_common_words(df[df["label"] == 0]["combined_text"]))

print("\nReal news:")
print(get_common_words(df[df["label"] == 1]["combined_text"]))

print("\nSaved graphs:")
print(RESULTS_DIR / "class_distribution.png")
print(RESULTS_DIR / "title_length_distribution.png")
print(RESULTS_DIR / "text_length_distribution.png")

print("\n✅ EDA completed successfully!")