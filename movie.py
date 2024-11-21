# Step 1: Importing Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 2: Load Dataset
df = pd.read_csv('C:\\Users\\ADMIN\\Desktop\\movie rating prediction using python\\dataset.csv')

# Step 3: Check the Column Names
print("Columns in dataset:", df.columns)

# Step 4: Basic Data Exploration
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nData Summary (Info):")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Clean Data (Handle missing values and non-numeric data)
# Handling missing values in 'Rating' column by filling with the mean
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

# Convert 'budget' and 'runtime' to numeric, coercing errors to NaN
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Duration'].fillna(df['Duration'].mean(), inplace=True)

# Handle non-numeric values in 'Votes' and convert them to numeric (removing commas, etc.)
df['Votes'] = df['Votes'].replace({',': ''}, regex=True)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Votes'].fillna(df['Votes'].mean(), inplace=True)

# Step 6: Encoding Categorical Variables
# If 'Genre' or 'Director' columns are categorical, encode them
# Check the exact column names in the dataset (case sensitivity)

df = pd.get_dummies(df, columns=['Genre', 'Director'], drop_first=True)

# Step 7: Feature Scaling (for numeric columns like 'Duration' and 'Votes')
scaler = StandardScaler()

# Scale 'Duration' and 'Votes' columns
df[['Duration', 'Votes']] = scaler.fit_transform(df[['Duration', 'Votes']])

# Step 8: Exploratory Data Analysis (EDA)
# Histogram of Ratings
sns.histplot(df['Rating'], kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Step 9: Feature Selection (if necessary)
# You can select specific features for your model, for example:
features = ['Duration', 'Votes'] + [col for col in df.columns if 'Genre' in col or 'Director' in col]
X = df[features]
y = df['Rating']

# Step 10: Model Building (if required, use machine learning models like linear regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict ratings on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 11: Conclusion
print("Model training and evaluation completed.")

