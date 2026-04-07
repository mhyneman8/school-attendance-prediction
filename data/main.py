from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Load CSV from same directory as this script (works from any cwd)
df = pd.read_csv("data/School_Attendance_dataset.csv")

# Clean column names
df.columns = (
    df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
)
# stnadardize student count and attendance rate column names
df.rename(columns={
    '2021_2022_student_count___year_to_date': '2021_2022_student_count',
    '2021_2022_attendance_rate___year_to_date': '2021_2022_attendance_rate',
}, inplace=True)

# "2021-2022 student count"
# → "20212022_student_count"

# handle missing values
df['category'] = df['category'].fillna('unknown')

# Count = 0 is reasonable if data is missing
# Attendance rate = average of existing rates because 0 would distort data

df.fillna({
    '2021_2022_student_count': 0,
    '2020_2021_attendance_rate': df['2020_2021_attendance_rate'].mean(),
    '2020_2021_student_count': 0,
    '2021_2022_attendance_rate': df['2021_2022_attendance_rate'].mean(),
    '2019_2020_student_count': 0,
    '2019_2020_attendance_rate': df['2019_2020_attendance_rate'].mean(),
}, inplace=True)

# Fix data types

# all student counts in integers
df['2021_2022_student_count'] = df['2021_2022_student_count'].astype(int)
df['2020_2021_student_count'] = df['2020_2021_student_count'].astype(int)
df['2019_2020_student_count'] = df['2019_2020_student_count'].astype(int)

# dates stored as strings - convert to datetime
df['date_update'] = pd.to_datetime(df['date_update'])

# Standardize categorical data
# convert to lowercase
df['student_group'] = df['student_group'].str.strip().str.lower()
df['category'] = df['category'].str.strip().str.lower()
df['district_name'] = df['district_name'].str.strip().str.lower()

# Normalize attendance rates
attendance_cols = [
    '2021_2022_attendance_rate',
    '2020_2021_attendance_rate',
    '2019_2020_attendance_rate',
]

scaler = MinMaxScaler()
df[attendance_cols] = scaler.fit_transform(df[attendance_cols])

# Normalize student counts
student_cols = [
    '2021_2022_student_count',
    '2020_2021_student_count',
    '2019_2020_student_count',
]

df[student_cols] = scaler.fit_transform(df[student_cols])

# save clearned data
df.to_csv('data/cleaned_data.csv', index=False)