import pandas as pd
from pathlib import Path

# Load CSV from same directory as this script (works from any cwd)
_csv_path = Path(__file__).resolve().parent / "School_Attendance_dataset.csv"
df = pd.read_csv(_csv_path)

print("--- head() ---")
print(df.head())

print("\n--- info() ---")
print(df.info())

print("\n--- describe() ---")
print(df.describe())

print("\n--- columns ---")
print(df.columns.tolist())

print("\n--- shape ---")
print(df.shape)