import matplotlib

# Use a non-GUI backend to avoid Tkinter errors
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Model performance values
# --------------------------------------------------

models = [
    "Logistic Regression",
    "Linear SVM",
    "Multinomial Naive Bayes"
]

accuracy = [98.99, 99.59, 96.24]
precision = [98.46, 99.41, 96.55]
recall = [99.69, 99.83, 96.51]
f1_score = [99.07, 99.62, 96.53]


# --------------------------------------------------
# 3. Configure bar positions
# --------------------------------------------------

x = np.arange(len(models))
width = 0.20


# --------------------------------------------------
# 4. Create the grouped bar chart
# --------------------------------------------------

plt.figure(figsize=(10, 6))

bars_accuracy = plt.bar(
    x - 1.5 * width,
    accuracy,
    width,
    label="Accuracy"
)

bars_precision = plt.bar(
    x - 0.5 * width,
    precision,
    width,
    label="Precision"
)

bars_recall = plt.bar(
    x + 0.5 * width,
    recall,
    width,
    label="Recall"
)

bars_f1 = plt.bar(
    x + 1.5 * width,
    f1_score,
    width,
    label="F1-Score"
)


# --------------------------------------------------
# 5. Add labels and formatting
# --------------------------------------------------

plt.title("Classification Model Performance Comparison")
plt.xlabel("Classification Model")
plt.ylabel("Score (%)")

plt.xticks(
    x,
    models,
    rotation=15
)

# All values are between 96% and 100%
plt.ylim(90, 101)

plt.legend()
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)


# --------------------------------------------------
# 6. Add values above each bar
# --------------------------------------------------

all_bars = [
    bars_accuracy,
    bars_precision,
    bars_recall,
    bars_f1
]

for bar_group in all_bars:
    for bar in bar_group:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.15,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )


# --------------------------------------------------
# 7. Save the figure
# --------------------------------------------------

plt.tight_layout()

output_path = FIGURE_DIR / "fig3_model_performance_comparison.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 3 generated successfully.")
print(f"Saved at: {output_path}")