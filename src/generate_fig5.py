import matplotlib

# Use a non-GUI backend to avoid Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Training time values
# --------------------------------------------------

models = [
    "Logistic Regression",
    "Linear SVM",
    "Multinomial Naive Bayes"
]

training_times = [
    0.51,
    0.37,
    0.02
]


# --------------------------------------------------
# 3. Create the bar chart
# --------------------------------------------------

plt.figure(figsize=(9, 5))

bars = plt.bar(
    models,
    training_times,
    edgecolor="black"
)

plt.title("Training Time Comparison of Classification Models")
plt.xlabel("Classification Model")
plt.ylabel("Training Time (seconds)")

plt.xticks(rotation=15)
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)


# --------------------------------------------------
# 4. Add values above the bars
# --------------------------------------------------

for bar, value in zip(bars, training_times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{value:.2f} s",
        ha="center",
        va="bottom",
        fontsize=10
    )


# --------------------------------------------------
# 5. Save the figure
# --------------------------------------------------

plt.tight_layout()

output_path = FIGURE_DIR / "fig4_training_time_comparison.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 4 generated successfully.")
print(f"Saved at: {output_path}")