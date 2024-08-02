# import pandas as pd
# import re
# from collections import Counter
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import nltk
# import json
# from functools import lru_cache

# nltk.download('stopwords')
# nltk.download('punkt')

# # Load stopwords
# stop_words = set(stopwords.words('english'))

# # Load JSON file with state names
# with open('states_data.json', 'r') as file:
#     states_data = json.load(file)

# # Access the list under the 'states info' key
# states_info = states_data['states info']

# # Extract state names from JSON data
# state_names = [state_info['State'].lower() for state_info in states_info]
# state_names = [name for name in state_names if name != 'code']

# # Compile a regular expression pattern for state names
# state_names_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(name) for name in state_names) + r')\b', re.IGNORECASE)

# # Load CSV file
# file_path = '3_govt_urls_state_only.csv'
# df = pd.read_csv(file_path)

# # Function to extract state names from text
# def extract_state_names(text):
#     return re.findall(state_names_pattern, text)

# # Function to preprocess text with caching
# @lru_cache(maxsize=None)
# def preprocess_text(text):
#     # Remove everything after '--' if present
#     if '--' in text:
#         text = text.split('--')[0]
#     text = text.strip()  # Remove leading and trailing spaces
    
#     # Extract and remove state names
#     state_names_extracted = extract_state_names(text)
#     text = state_names_pattern.sub('', text)
    
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
#     # Tokenize text
#     words = word_tokenize(text)
    
#     # Remove stopwords
#     words = [word for word in words if word not in stop_words]
    
#     return ' '.join(words), state_names_extracted

# # Function to generate n-grams
# def generate_ngrams(text, n):
#     tokens = word_tokenize(text)
#     ngrams = zip(*[tokens[i:] for i in range(n)])
#     return [' '.join(gram) for gram in ngrams]

# # Function to tokenize and count word frequencies with n-grams
# def get_most_common_ngrams(texts, n, min_freq=2):
#     all_ngrams = []
#     for text in texts:
#         processed_text, _ = preprocess_text(text)
#         ngrams = generate_ngrams(processed_text, n)
#         all_ngrams.extend(ngrams)
    
#     ngram_counts = Counter(all_ngrams)
#     # Filter out n-grams with frequency less than min_freq
#     filtered_ngrams = {ngram: freq for ngram, freq in ngram_counts.items() if freq >= min_freq}
#     return filtered_ngrams

# # Extract most common n-grams (bigrams and trigrams) from the 'Note' column
# min_freq = 2  # Minimum frequency for n-grams to be included
# bigrams = get_most_common_ngrams(df['Note'], 2, min_freq)
# trigrams = get_most_common_ngrams(df['Note'], 3, min_freq)

# # Combine bigrams and trigrams into a single dictionary
# most_common_ngrams = {**bigrams, **trigrams}

# # Create a list of the most common n-grams
# most_common_ngrams_list = list(most_common_ngrams.keys())

# # Function to create a dynamic sector mapping
# def create_dynamic_sector_mapping(common_ngrams):
#     # Placeholder for dynamic sector keywords mapping
#     sector_keywords = {}
    
#     #dynamically determine sector categories
#     for ngram in common_ngrams:
#         sector = ngram.lower()  # Treat each common n-gram as a potential sector
#         if sector not in sector_keywords:
#             sector_keywords[sector] = []
#         sector_keywords[sector].append(ngram)
    
#     return sector_keywords

# # Create dynamic sector keywords mapping
# sector_keywords = create_dynamic_sector_mapping(most_common_ngrams_list)

# # Function to determine sector for a given note
# def determine_sector(note):
#     processed_note, _ = preprocess_text(note)
#     ngrams = generate_ngrams(processed_note, 2) + generate_ngrams(processed_note, 3)
#     for ngram in ngrams:
#         for sector, keywords in sector_keywords.items():
#             if ngram in keywords:
#                 return sector
#     return 'Unknown'

# # Apply the determine_sector function to the 'Note' column and extract state names
# df['Processed_Note'], df['States'] = zip(*df['Note'].apply(preprocess_text))
# df['Sector'] = df['Note'].apply(determine_sector)

# # Reorder columns to place 'Sector' and 'Extracted_States' first
# df = df[['Sector', 'States', 'Domain', 'Note']]

# # Save the result to a new CSV file
# df.to_csv('classified_data.csv', index=False)

# # Display the result
# print(df)
import pandas as pd
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
from functools import lru_cache

nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load JSON file with state names
with open('states_data.json', 'r') as file:
    states_data = json.load(file)

# Access the list under the 'states info' key
states_info = states_data['states info']

# Extract state names from JSON data
state_names = [state_info['State'].lower() for state_info in states_info]
state_names = [name for name in state_names if name != 'code']

# Compile a regular expression pattern for state names
state_names_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(name) for name in state_names) + r')\b', re.IGNORECASE)

# Load CSV file
file_path = '3_govt_urls_state_only.csv'
df = pd.read_csv(file_path)

# Function to extract state names from text
def extract_state_names(text):
    found_states = re.findall(state_names_pattern, text)
    # Convert to lowercase and remove duplicates
    unique_states = list(set(state.lower() for state in found_states))
    return unique_states

# Function to preprocess text with caching
@lru_cache(maxsize=None)
def preprocess_text(text):
    # Remove everything after '--' if present
    if '--' in text:
        text = text.split('--')[0]
    text = text.strip()  # Remove leading and trailing spaces
    
    # Extract and remove state names
    state_names_extracted = extract_state_names(text)
    text = state_names_pattern.sub('', text)
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words), state_names_extracted

# Function to generate n-grams
def generate_ngrams(text, n):
    tokens = word_tokenize(text)
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(gram) for gram in ngrams]

# Function to tokenize and count word frequencies with n-grams
def get_most_common_ngrams(texts, n, min_freq=2):
    all_ngrams = []
    for text in texts:
        processed_text, _ = preprocess_text(text)
        ngrams = generate_ngrams(processed_text, n)
        all_ngrams.extend(ngrams)
    
    ngram_counts = Counter(all_ngrams)
    # Filter out n-grams with frequency less than min_freq
    filtered_ngrams = {ngram: freq for ngram, freq in ngram_counts.items() if freq >= min_freq}
    return filtered_ngrams

# Extract most common n-grams (bigrams and trigrams) from the 'Note' column
min_freq = 2  # Minimum frequency for n-grams to be included
bigrams = get_most_common_ngrams(df['Note'], 2, min_freq)
trigrams = get_most_common_ngrams(df['Note'], 3, min_freq)

# Combine bigrams and trigrams into a single dictionary
most_common_ngrams = {**bigrams, **trigrams}

# Create a list of the most common n-grams
most_common_ngrams_list = list(most_common_ngrams.keys())

# Function to create a dynamic sector mapping
def create_dynamic_sector_mapping(common_ngrams):
    # Placeholder for dynamic sector keywords mapping
    sector_keywords = {}
    
    # dynamically determine sector categories
    for ngram in common_ngrams:
        sector = ngram.lower()  # Treat each common n-gram as a potential sector
        if sector not in sector_keywords:
            sector_keywords[sector] = []
        sector_keywords[sector].append(ngram)
    
    return sector_keywords

# Create dynamic sector keywords mapping
sector_keywords = create_dynamic_sector_mapping(most_common_ngrams_list)

# Function to determine sector for a given note
def determine_sector(note):
    processed_note, _ = preprocess_text(note)
    ngrams = generate_ngrams(processed_note, 2) + generate_ngrams(processed_note, 3)
    for ngram in ngrams:
        for sector, keywords in sector_keywords.items():
            if ngram in keywords:
                return sector
    return 'Unknown'

# Apply the determine_sector function to the 'Note' column and extract state names
df['Processed_Note'], df['States'] = zip(*df['Note'].apply(preprocess_text))
df['Sector'] = df['Note'].apply(determine_sector)

# Reorder columns to place 'Sector' and 'Extracted_States' first
df = df[['Sector', 'States', 'Domain', 'Note']]

# Save the result to a new CSV file
df.to_csv('classified_data.csv', index=False)

# Display the result
print(df)
