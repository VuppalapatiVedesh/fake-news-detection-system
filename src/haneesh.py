import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output folder
OUTPUT_DIR = Path("results/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# REPLACE THESE VALUES WITH YOUR ACTUAL RESULTS
# ---------------------------------------------------------

features = [
    "Feature 1",
    "Feature 2",
    "Feature 3",
    "Feature 4",
    "Feature 5"
]

real_data = [
    0.82,
    0.76,
    0.91,
    0.68,
    0.79
]

synthetic_data = [
    0.79,
    0.73,
    0.87,
    0.65,
    0.75
]

# ---------------------------------------------------------
# CREATE GRAPH
# ---------------------------------------------------------

x = np.arange(len(features))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    real_data,
    width,
    label="Real Data"
)

plt.bar(
    x + width / 2,
    synthetic_data,
    width,
    label="Synthetic Data"
)

plt.xlabel("Features")
plt.ylabel("Similarity")
plt.title("Real vs Synthetic Feature Distribution")

plt.xticks(x, features)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure1_feature_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()