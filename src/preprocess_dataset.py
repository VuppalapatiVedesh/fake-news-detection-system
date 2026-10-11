import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

fake_path = DATA_DIR / "Fake.csv"
true_path = DATA_DIR / "True.csv"

# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# --------------------------------------------------
# 3. Assign labels
# --------------------------------------------------

fake_df["label"] = 0
true_df["label"] = 1

# --------------------------------------------------
# 4. Combine both datasets
# --------------------------------------------------

df = pd.concat([fake_df, true_df], ignore_index=True)

# --------------------------------------------------
# 5. Keep only required columns
# --------------------------------------------------

df = df[["title", "text", "subject", "date", "label"]]

# --------------------------------------------------
# 6. Remove missing text
# --------------------------------------------------

df = df.dropna(subset=["title", "text"])

# --------------------------------------------------
# 7. Combine title and article text
# --------------------------------------------------

df["combined_text"] = (
    df["title"].astype(str) + " " + df["text"].astype(str)
)

# --------------------------------------------------
# 8. Remove duplicate articles
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates(subset=["combined_text"])

after_duplicates = len(df)

# --------------------------------------------------
# 9. Shuffle the dataset
# --------------------------------------------------

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --------------------------------------------------
# 10. Save processed dataset
# --------------------------------------------------

output_path = PROCESSED_DIR / "news_dataset.csv"

df.to_csv(output_path, index=False)

# --------------------------------------------------
# 11. Display summary
# --------------------------------------------------

print("========== PREPROCESSING SUMMARY ==========")

print("Original total articles:", before_duplicates)
print("Articles after duplicate removal:", after_duplicates)
print("Duplicates removed:", before_duplicates - after_duplicates)

print("\nFinal dataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())

print(f"\n✅ Processed dataset saved to: {output_path}")