import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.model_selection import train_test_split

# Set Seaborn style
sns.set_style('whitegrid')

# Load the CSV file into a DataFrame with proper column names
df = pd.read_csv('all-data.csv', encoding="ISO-8859-1", names=['sentiment', 'text'])

# Display the first few rows of the DataFrame
print(df.head())

# Display DataFrame information (data types, non-null values, etc.)
print(df.info())

# Check for missing values in each column
print(df.isna().sum())

# Display the count of each sentiment category
print(df['sentiment'].value_counts())

# Extract the text and sentiment data
texts = df['text'].values
sentiments = df['sentiment'].values

# Print the shapes of the arrays
print(f'Shape of texts: {texts.shape}')
print(f'Shape of sentiments: {sentiments.shape}')

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(texts, sentiments, test_size=0.4, random_state=42)

# Print the shapes of the training and testing sets
print(f'Shape of x_train: {x_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of x_test: {x_test.shape}')
print(f'Shape of y_test: {y_test.shape}')

# Create DataFrames from training data
df1 = pd.DataFrame(x_train, columns=['news'])
df2 = pd.DataFrame(y_train, columns=['sentiment'])

# Concatenate DataFrames along the columns
df_train = pd.concat([df1, df2], axis=1)

# Display the first few rows of the concatenated DataFrame
print(df_train.head())

# Create DataFrames from test data
df3 = pd.DataFrame(x_test, columns=['news'])
df4 = pd.DataFrame(y_test, columns=['sentiment'])

# Concatenate DataFrames along the columns
df_test = pd.concat([df3, df4], axis=1)

# Display the first few rows of the concatenated DataFrame
print(df_test.head())

# Function to remove punctuation
def remove_punctuation(text):
    if isinstance(text, float):
        return text
    return ''.join([char for char in text if char not in string.punctuation])

# Apply the function to remove punctuation from the 'news' column in the train dataset
df_train['news'] = df_train['news'].apply(remove_punctuation)

# Apply the function to remove punctuation from the 'news' column in the test dataset
df_test['news'] = df_test['news'].apply(remove_punctuation)

# Display the first few rows of the train dataset to confirm punctuation removal
print("Train DataFrame after removing punctuation:")
print(df_train.head())

# Display the first few rows of the test dataset to confirm punctuation removal
print("Test DataFrame after removing punctuation:")
print(df_test.head())

# Function to compute N-grams
def compute_ngrams(text, n):
    tokens = word_tokenize(text)
    
    # Generate N-grams
    n_grams = ngrams(tokens, n)
    
    return list(n_grams)

#Compute bigrams for the cleaned text in train dataset
df_train['bigrams'] = df_train['news'].apply(lambda x: compute_ngrams(x, 2))

#Compute bigrams for the cleaned text in test dataset
df_test['bigrams'] = df_test['news'].apply(lambda x: compute_ngrams(x, 2))

# Display the first few rows of the train dataset with bigrams
print("Train DataFrame with bigrams:")
print(df_train.head())

# Display the first few rows of the test dataset with bigrams
print("Test DataFrame with bigrams:")
print(df_test.head())
