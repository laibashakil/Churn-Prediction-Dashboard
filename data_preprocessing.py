import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill missing values in 'TotalCharges' with the median
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert to numeric
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert binary categorical variables (e.g., gender)
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# One-Hot Encoding for other categorical features
df = pd.get_dummies(df, columns=['InternetService', 'PaymentMethod', 'Contract', 'MultipleLines', 'OnlineSecurity', 
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                 'Dependents', 'Partner', 'PhoneService', 'PaperlessBilling'], drop_first=True)

# Drop irrelevant columns (e.g., customerID)
df = df.drop(['customerID'], axis=1)

# Print the first few rows after preprocessing
print("\nFirst 5 rows after preprocessing:")
print(df.head())

# List of numerical features to scale
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Initialize the scaler
scaler = StandardScaler()

# Scale the numerical features
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the train and test sets
print(f'\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}')

# Save the feature list (after get_dummies) to a file for future use
feature_list = X.columns.tolist()

# Save the feature list as a pickle file
import joblib
joblib.dump(feature_list, 'feature_list.pkl')

