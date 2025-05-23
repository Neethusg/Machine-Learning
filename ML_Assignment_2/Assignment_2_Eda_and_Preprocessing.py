# Assignment_2_EDA_and_Preprocessing


from google.colab import files
uploaded = files.upload()

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("Employee.csv")

#Q1.Explore the data, list down the unique values in each feature and find its length.Perform the statistical analysis and renaming of the columns.
print("First 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nShape of data:", df.shape)

df.info()

df.head(10)

print("\nUnique values in each column:")
for col in df.columns:
    unique_val = df[col].unique()
    print(f"{col} -> {unique_val} (Count: {len(unique_val)})")

print("\nStatistical Summary:")
print(df.describe(include='all'))

df_renamed = df.rename(columns={'Company': 'Company name', 'Age': 'Age of Employees'})

print("\nRenamed Columns:")
print(df_renamed.columns)

df_renamed.head(10)

#Q2. Data Cleaning
#1. Find the missing and inappropriate values, treat them appropriately.
# Checking the missing values
print("Missing values per column:\n", df.isnull().sum())

df['Place'].fillna('Unknown', inplace=True)
df['Company'].fillna('Unknown', inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)

median_salary = df['Salary'].median()
df['Salary'].fillna(median_salary, inplace=True)

print("Missing values per column:\n", df.isnull().sum())

#2. Duplicate Rows
print("Duplicate rows:\n", df[df.duplicated()])

No_Duplicates = df.drop_duplicates()
print(No_Duplicates)

#3. Find the outliers.
#IQR Method
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]

print("Outliers using IQR method:\n", outliers)

#Z-Score method
z_scores = np.abs(stats.zscore(df['Salary']))
outliers_z = df[z_scores > 3]
print("Outliers using Z-score method:\n", outliers_z)

#4. Replace the value 0 in age as NaN
df['Age'].replace(0, np.nan, inplace=True)

print("\nUpdated DataFrame with 0s in age replaced by NaN:")
print(df)

#5. Treat the null values in all columns using any measures(removing/ replace the values with mean/median/mode)
df.fillna(df.median(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\n cleaned DataFrame:")
print(df)

#Q3. Data Analysis
#1. Filter the data with age >40 and salary<5000
filtered_df = df[(df['Age'] > 40) & (df['Salary'] < 5000)]
print("\nFiltered Data (Age > 40 and Salary < 5000):")
print(filtered_df)

#2. Plot the chart with age and salary
plt.figure(figsize=(8, 5))
plt.bar(df['Age'], df['Salary'], color='skyblue')
plt.title('Salary by Age')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#3. Count the number of people from each place and represent it visually
place_counts = df['Place'].value_counts()
place_counts

place_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, figsize=(6,6), colors=['#66b3ff','#ff9999','#99ff99','#ffcc99'])
plt.title('People Distribution by Place')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Bar Chart
plt.figure(figsize=(7, 5))
place_counts.plot(kind='bar', color='coral')
plt.title('Number of People from Each Place')
plt.xlabel('Place')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Q4. Data Encoding
#  Label Encoding (for gender)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])
df['Gender_encoded']

#  One-Hot Encoding (for place)
df_encoded = pd.get_dummies(df, columns=['Place'], prefix='place')

print("\nEncoded DataFrame:")
print(df_encoded)

#Q5. Feature Scaling
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
df_encoded = df.drop(['Company', 'Gender', 'Place', 'Country'], axis=1)
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

# Apply StandardScaler
standard_scaled = pd.DataFrame(
    scaler_standard.fit_transform(df_encoded),
    columns=[f"{col}_std" for col in df_encoded.columns]
)

# Apply MinMaxScaler
minmax_scaled = pd.DataFrame(
    scaler_minmax.fit_transform(df_encoded),
    columns=[f"{col}_minmax" for col in df_encoded.columns]
)
standard_scaled

minmax_scaled

scaled_df = pd.concat([standard_scaled, minmax_scaled], axis=1)

print("\nScaled Data (Standard & MinMax):")
print(scaled_df)

