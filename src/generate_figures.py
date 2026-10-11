import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
df = pd.read_csv("data/processed/news_dataset.csv")

# Calculate word counts
df["text_length"] = df["combined_text"].fillna("").str.split().str.len()

# Convert labels
df["label_name"] = df["label"].map({
    0: "Fake News",
    1: "Real News"
})

# Plot distributions
plt.figure(figsize=(8, 5))

plt.hist(
    df[df["label_name"] == "Fake News"]["text_length"],
    bins=50,
    alpha=0.6,
    label="Fake News"
)

plt.hist(
    df[df["label_name"] == "Real News"]["text_length"],
    bins=50,
    alpha=0.6,
    label="Real News"
)

plt.title("Distribution of Article Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(
    "fig2_text_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()