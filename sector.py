import pandas as pd
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
from functools import lru_cache

def main():
    # Download necessary NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Load JSON file with state names
    with open('states_data.json', 'r') as file:
        states_data = json.load(file)

    # Access the list under the 'states info' key
    states_info = states_data['states info']

    # Extract state names from JSON data and store in a set for O(1) lookups
    state_names_set = {state_info['State'].lower() for state_info in states_info}

    # Load CSV file
    file_path = '3_govt_urls_state_only.csv'
    df = pd.read_csv(file_path)

    # Function to preprocess text with caching
    @lru_cache(maxsize=None)
    def preprocess_text(text):
        # Remove everything after '--' if present
        if '--' in text:
            text = text.split('--')[0]
        text = text.strip()  # Remove leading and trailing spaces
        
        # Tokenize text and remove state names
        tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
        state_names_extracted = {token for token in tokens if token in state_names_set}
        tokens = [token for token in tokens if token not in state_names_set]
        
        # Further text processing
        tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]  # Remove punctuation
        tokens = [token for token in tokens if token and token not in stop_words]  # Remove stopwords
        
        return ' '.join(tokens), list(state_names_extracted)

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
    most_common_ngrams_list = list(most_common_ngrams.keys())

    # Function to create a dynamic sector mapping
    def create_dynamic_sector_mapping(common_ngrams):
        # Placeholder for dynamic sector keywords mapping
        sector_keywords = {}
        
        # Assign a generic sector or categorize based on additional logic
        for ngram in common_ngrams:
            sector = ngram.lower()  # Treat each common n-gram as a potential sector
            if sector not in sector_keywords:
                sector_keywords[sector] = []
            sector_keywords[sector].append(ngram)
        
        return sector_keywords

    # Create dynamic sector keywords mapping
    sector_keywords = create_dynamic_sector_mapping(most_common_ngrams_list)

    # Function to determine sector for a given note
    def determine_sector(preprocessed_text):
        ngrams = generate_ngrams(preprocessed_text, 2) + generate_ngrams(preprocessed_text, 3)
        for ngram in ngrams:
            for sector, keywords in sector_keywords.items():
                if ngram in keywords:
                    return sector
        return 'Unknown'

    # Apply the preprocessing and sector determination to the 'Note' column
    df['Processed_Note'], df['States'] = zip(*df['Note'].apply(preprocess_text))
    df['Sector'] = df['Processed_Note'].apply(determine_sector)

    # Reorder columns to place 'Sector' and 'Extracted_States' first
    df = df[['Sector', 'States', 'Domain', 'Note']]

    # Save the result to a new CSV file
    df.to_csv('classified_data.csv', index=False)

    # Display the result
    print(df)

if __name__ == "__main__":
    main()
