import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define dataset paths
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

fake_path = DATA_DIR / "Fake.csv"
true_path = DATA_DIR / "True.csv"

# --------------------------------------------------
# 2. Check whether files exist
# --------------------------------------------------

if not fake_path.exists():
    raise FileNotFoundError(f"Missing file: {fake_path}")

if not true_path.exists():
    raise FileNotFoundError(f"Missing file: {true_path}")

# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

print("Fake news dataset shape:", fake_df.shape)
print("Real news dataset shape:", true_df.shape)

# --------------------------------------------------
# 4. Display column names
# --------------------------------------------------

print("\nFake dataset columns:")
print(fake_df.columns.tolist())

print("\nReal dataset columns:")
print(true_df.columns.tolist())

# --------------------------------------------------
# 5. Display first few rows
# --------------------------------------------------

print("\nFirst 3 fake news articles:")
print(fake_df.head(3))

print("\nFirst 3 real news articles:")
print(true_df.head(3))

# --------------------------------------------------
# 6. Check missing values
# --------------------------------------------------

print("\nMissing values in Fake dataset:")
print(fake_df.isnull().sum())

print("\nMissing values in Real dataset:")
print(true_df.isnull().sum())

# --------------------------------------------------
# 7. Check duplicate rows
# --------------------------------------------------

print("\nDuplicate rows in Fake dataset:", fake_df.duplicated().sum())
print("Duplicate rows in Real dataset:", true_df.duplicated().sum())