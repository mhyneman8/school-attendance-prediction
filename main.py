import runpy

print("1. Clean data")
runpy.run_path("data/cleaning_data.py")

print("2. Categorize data")
runpy.run_path("data/categorize_data.py")

print("3. Run Linear Regression model")
runpy.run_path("data/linear_regression.py")
