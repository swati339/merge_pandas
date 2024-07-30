import re
from nltk import ngrams
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already done
nltk.download('stopwords')

# Original sentence
sentence = 'I reside in Bengaluru.'
n = 1

# Generate unigrams
unigrams = ngrams(sentence.split(), n)
for grams in unigrams:
    print(grams)

# Tokenization and Stopword Filtering


text = "I am learning python and enjoying it. It is very important for the data scientists."

# Remove non-alphanumeric characters and convert to lowercase
text_cleaned = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

# Split the cleaned text into words
words = text_cleaned.split()

# Print the list of words
print("All words:", words)

# Set of English stopwords
stop_words = set(stopwords.words('english'))

# Print the list of stopwords
print("Stopwords:", stopwords.words('english'))

# Filter out the stopwords from the list of words
filtered_words = [word for word in words if word not in stop_words]

# Print the filtered words
print("Filtered words:", filtered_words)
