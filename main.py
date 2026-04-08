import pandas as pd

# Load CSV from same directory as this script (works from any cwd)
df = pd.read_csv("data/cleaned_data.csv")

print(df.head())

print(df.columns.tolist())
print('is null')
print(df.isnull().sum())
print(df['2021_2022_attendance_rate'])

print('duplicates')
print(df.duplicated().sum())

print(df.info())