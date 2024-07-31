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

# Ensure NLTK data packages are downloaded
nltk.download('punkt')

def process_text_data(file_path, test_size=0.4, random_state=42, ngram_n=2):
    # Load the CSV file into a DataFrame with proper column names
    df = pd.read_csv(file_path, encoding="ISO-8859-1", names=['sentiment', 'text'])

    # Display the first few rows of the DataFrame
    print("Initial DataFrame head:")
    print(df.head())

    # Display DataFrame information (data types, non-null values, etc.)
    print("DataFrame info:")
    print(df.info())

    # Check for missing values in each column
    print("Missing values in each column:")
    print(df.isna().sum())

    # Display the count of each sentiment category
    print("Sentiment value counts:")
    print(df['sentiment'].value_counts())

    # Extract the text and sentiment data
    texts = df['text'].values
    sentiments = df['sentiment'].values

    # Print the shapes of the arrays
    print(f'Shape of texts: {texts.shape}')
    print(f'Shape of sentiments: {sentiments.shape}')

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(texts, sentiments, test_size=test_size, random_state=random_state)

    # Print the shapes of the training and testing sets
    print(f'Shape of x_train: {x_train.shape}')
    print(f'Shape of y_train: {y_train.shape}')
    print(f'Shape of x_test: {x_test.shape}')
    print(f'Shape of y_test: {y_test.shape}')

    # Create DataFrames from training data
    df_train = pd.DataFrame({'news': x_train, 'sentiment': y_train})
    df_test = pd.DataFrame({'news': x_test, 'sentiment': y_test})

    # Function to remove punctuation
    def remove_punctuation(text):
        if isinstance(text, float):
            return text
        return ''.join([char for char in text if char not in string.punctuation])

    # Apply the function to remove punctuation from the 'news' column in the train and test datasets
    df_train['news'] = df_train['news'].apply(remove_punctuation)
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
        n_grams = ngrams(tokens, n)
        return list(n_grams)

    # Compute bigrams (or N-grams) for the cleaned text in train and test datasets
    df_train['ngrams'] = df_train['news'].apply(lambda x: compute_ngrams(x, ngram_n))
    df_test['ngrams'] = df_test['news'].apply(lambda x: compute_ngrams(x, ngram_n))

    # Display the first few rows of the train dataset with N-grams
    print("Train DataFrame with N-grams:")
    print(df_train.head())

    # Display the first few rows of the test dataset with N-grams
    print("Test DataFrame with N-grams:")
    print(df_test.head())

    return df_train, df_test

# Example usage
train_data, test_data = process_text_data('all-data.csv', test_size=0.4, random_state=42, ngram_n=2)
