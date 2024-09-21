import pandas as pd

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Get some basic statistics
print(df.describe())
