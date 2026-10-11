import matplotlib

# Use a non-GUI backend to avoid Tkinter errors
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Define the Linear SVM confusion matrix
# --------------------------------------------------

confusion_matrix = np.array([
    [3557, 25],
    [7, 4232]
])

labels = [
    "Fake News",
    "Real News"
]


# --------------------------------------------------
# 3. Create the confusion matrix heatmap
# --------------------------------------------------

plt.figure(figsize=(7, 6))

sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor="white",
    cbar=True,
    annot_kws={"size": 14}
)

plt.title("Confusion Matrix of Linear SVM", fontsize=14)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("Actual Label", fontsize=11)


# --------------------------------------------------
# 4. Save the figure
# --------------------------------------------------

plt.tight_layout()

output_path = FIGURE_DIR / "fig4_linear_svm_confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 4 generated successfully.")
print(f"Saved at: {output_path}")