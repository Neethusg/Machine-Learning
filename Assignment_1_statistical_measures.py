ASSIGNMENT_1_Statistical Measures

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from google.colab import files
uploaded = files.upload()


#Q1. Perform basic EDA
#Loading dataset

df = pd.read_csv('house_price.csv')

df.info()

df.head()

print(df.info())
print(df.shape)

print(df.isnull().sum())

print(df.describe())

print(df['location'].value_counts().head(10))

print(df['size'].value_counts())

# Destribution of price
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True)
plt.title('Price Distribution')
plt.show()

# BHK count
sns.countplot(x='bhk', data=df)
plt.title('BHK Count')
plt.show()

# Detecting Outliers
sns.boxplot(x=df['price'])
plt.title('Boxplot of Price')
plt.show()


# Q2. Detect the outliers using following methods and remove it using methods like trimming / capping/ imputation using mean or median
# a) Mean and Standard deviation

mean = df['price_per_sqft'].mean()
std = df['price_per_sqft'].std()

lower_limit = mean - 3 * std
upper_limit = mean + 3 * std

# Remove outliers (trimming)
df_trimmed_mean_std = df[(df['price_per_sqft'] >= lower_limit) & (df['price_per_sqft'] <= upper_limit)]

df_trimmed_mean_std['price_per_sqft']

#b) Percentile Method (5th – 95th Percentile)

lower_percentile = df['price_per_sqft'].quantile(0.05)
upper_percentile = df['price_per_sqft'].quantile(0.95)

# Trimming outliers
df_trimmed_percentile = df[(df['price_per_sqft'] >= lower_percentile) & (df['price_per_sqft'] <= upper_percentile)]
df_trimmed_percentile

# Capping

df_capped_percentile = df.copy()
df_capped_percentile['price_per_sqft'] = df['price_per_sqft'].clip(lower_percentile, upper_percentile)

df_capped_percentile['price_per_sqft']

#c) IQR (Inter Quartile Range) Method

Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Trimming outliers
df_trimmed_iqr = df[(df['price_per_sqft'] >= lower_bound) & (df['price_per_sqft'] <= upper_bound)]
df_trimmed_iqr

# d) Z-Score Method

import scipy.stats as stats
z_scores =stats.zscore(df['price_per_sqft'])

# Trimming where |z| < 3
df_trimmed_zscore = df[(np.abs(z_scores) < 3)]

df_trimmed_zscore


#Q3. Create a box plot and use this to determine which method seems to work best to remove outliers for this data
plt.figure(figsize=(12, 6))
sns.boxplot(data=[
    df['price_per_sqft'],
    df_trimmed_mean_std['price_per_sqft'],
    df_trimmed_percentile['price_per_sqft'],
    df_trimmed_iqr['price_per_sqft'],
    df_trimmed_zscore['price_per_sqft']
])
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[
    'Original',
    'Mean ± 3*Std',
    'Percentile (5–95%)',
    'IQR',
    'Z-Score'
])
plt.title('Boxplot Comparison of Outlier Removal Methods')
plt.ylabel('Price per Sqft')
plt.grid(True)
plt.tight_layout()
plt.show()


#Q4. Draw histplot to check the normality of the column(price per sqft column) and perform transformations if needed. Check the skewness and kurtosis before and after the transformation.

from scipy.stats import skew, kurtosis
sns.histplot(df['price_per_sqft'], kde=True)
plt.title("Original Distribution")
plt.show()

print("Skewness:", skew(df['price_per_sqft']))
print("Kurtosis:", kurtosis(df['price_per_sqft']))

# Apply log transformation (or Box-Cox if needed)
df['price_per_sqft_log'] = np.log1p(df['price_per_sqft'])

# After transformation
sns.histplot(df['price_per_sqft_log'], kde=True)
plt.title("Log Transformed Distribution")
plt.show()

print("Skewness (log):", skew(df['price_per_sqft_log']))
print("Kurtosis (log):", kurtosis(df['price_per_sqft_log'])


#Q5. Check the correlation between all the numerical columns and plot heatmap.

numerical_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numerical_cols].corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

numerical_df = df.select_dtypes(include=['int64', 'float64'])
corr = numerical_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


#Q6. Draw Scatter plot between the variables to check the correlation between them.

sns.scatterplot(data=df, x='price_per_sqft', y='total_sqft')
plt.title('Price per Sqft vs Total Sqft')
plt.show()

sns.scatterplot(data=df, x='price_per_sqft', y='bhk')
plt.title('Price per Sqft vs bhk')
plt.show()

sns.scatterplot(data=df, x='price_per_sqft', y='bath')
plt.title('Price per Sqft vs bath')
plt.show()

sns.scatterplot(data=df, x='price_per_sqft', y='bhk')
plt.title('Price per Sqft vs price')
plt.show()

