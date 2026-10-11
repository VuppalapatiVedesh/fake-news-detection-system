import matplotlib

# Use a non-GUI backend to avoid Tkinter errors
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "news_dataset.csv"
FIGURE_DIR = BASE_DIR / "figures"

# Create the figures folder if it does not exist
FIGURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load the processed dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Calculate article length
# --------------------------------------------------

# Count the number of words in combined_text
df["text_length"] = (
    df["combined_text"]
    .fillna("")
    .str.split()
    .str.len()
)

# Convert numerical labels into readable names
df["label_name"] = df["label"].map({
    0: "Fake News",
    1: "Real News"
})


# --------------------------------------------------
# 4. Separate fake and real news article lengths
# --------------------------------------------------

fake_lengths = df[
    df["label_name"] == "Fake News"
]["text_length"]

real_lengths = df[
    df["label_name"] == "Real News"
]["text_length"]


# --------------------------------------------------
# 5. Create the histogram
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    fake_lengths,
    bins=50,
    alpha=0.60,
    label="Fake News",
    edgecolor="black"
)

plt.hist(
    real_lengths,
    bins=50,
    alpha=0.60,
    label="Real News",
    edgecolor="black"
)

plt.title("Distribution of Article Lengths by News Category")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")

plt.legend()
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()


# --------------------------------------------------
# 6. Save the figure
# --------------------------------------------------

output_path = FIGURE_DIR / "fig2_text_length_distribution.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure 2 saved successfully at:")
print(output_path)