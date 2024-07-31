import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split

# Set Seaborn style
sns.set_style('whitegrid')

# Function to remove punctuation
def remove_punctuation(text):
    if isinstance(text, float):
        return text
    return ''.join([char for char in text if char not in string.punctuation])

# Custom function to tokenize text into words
def tokenize(text):
    return text.split()

# Custom function to compute N-grams
def custom_ngrams(tokens, n):
    n_grams = []
    for i in range(len(tokens) - n + 1):
        n_grams.append(tuple(tokens[i:i + n]))
    return n_grams

# Custom function to compute bigrams
def compute_bigrams(tokens):
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append((tokens[i], tokens[i + 1]))
    return bigrams

# Main function to process text data
def process_text_data(file_path, test_size=0.4, random_state=42):
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
    df1 = pd.DataFrame({'news': x_train, 'sentiment': y_train})
    df2 = pd.DataFrame({'news': x_test, 'sentiment': y_test})

    # Apply the function to remove punctuation from the 'news' column in the train and test datasets
    df1['news'] = df1['news'].apply(remove_punctuation)
    df2['news'] = df2['news'].apply(remove_punctuation)

    # Display the first few rows of the train dataset to confirm punctuation removal
    print("Train DataFrame after removing punctuation:")
    print(df1.head())

    # Display the first few rows of the test dataset to confirm punctuation removal
    print("Test DataFrame after removing punctuation:")
    print(df2.head())

    return df1, df2

# Example usage of the main function
train_data, test_data = process_text_data('all-data.csv', test_size=0.4, random_state=42)

# Example usage of the custom N-gram and bigram functions
train_data['ngrams'] = train_data['news'].apply(lambda x: custom_ngrams(tokenize(x), 2))
train_data['bigrams'] = train_data['news'].apply(lambda x: compute_bigrams(tokenize(x)))
test_data['ngrams'] = test_data['news'].apply(lambda x: custom_ngrams(tokenize(x), 2))
test_data['bigrams'] = test_data['news'].apply(lambda x: compute_bigrams(tokenize(x)))

# Display the first few rows of the train dataset with bigrams
print("Train DataFrame with bigrams:")
print(train_data.head())

# Display the first few rows of the test dataset with bigrams
print("Test DataFrame with bigrams:")
print(test_data.head())
