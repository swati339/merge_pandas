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

def load_stopwords():
    return set(stopwords.words('english'))

def load_state_names(filename):
    with open(filename, 'r') as file:
        states_data = json.load(file)
    states_info = states_data['states info']
    state_names = [state_info['State'].lower() for state_info in states_info]
    return set(state_names)

def preprocess_text(text, state_names, stop_words):
    # Remove everything after '--' if present
    if '--' in text:
        text = text.split('--')[0]
    text = text.strip()  # Remove leading and trailing spaces
    
    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords and state names
    words = [word for word in words if word not in stop_words and word not in state_names]
    
    return ' '.join(words)

def generate_ngrams(text, n):
    tokens = word_tokenize(text)
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(gram) for gram in ngrams]

def get_most_common_ngrams(texts, state_names, stop_words, n, min_freq=2):
    all_ngrams = []
    for text in texts:
        processed_text = preprocess_text(text, state_names, stop_words)
        ngrams = generate_ngrams(processed_text, n)
        all_ngrams.extend(ngrams)
    
    ngram_counts = Counter(all_ngrams)
    # Filter out n-grams with frequency less than min_freq
    filtered_ngrams = {ngram: freq for ngram, freq in ngram_counts.items() if freq >= min_freq}
    return filtered_ngrams

def create_dynamic_sector_mapping(common_ngrams):
    sector_keywords = {}
    for ngram in common_ngrams:
        sector = ngram.lower()
        if sector not in sector_keywords:
            sector_keywords[sector] = []
        sector_keywords[sector].append(ngram)
    return sector_keywords

def determine_sector(note, state_names, stop_words, sector_keywords):
    processed_note = preprocess_text(note, state_names, stop_words)
    ngrams = generate_ngrams(processed_note, 2) + generate_ngrams(processed_note, 3)
    for ngram in ngrams:
        for sector, keywords in sector_keywords.items():
            if ngram in keywords:
                return sector
    return 'Unknown'

def extract_states_from_text(text, state_names):
    text = text.lower()
    found_states = set()
    for state in state_names:
        #multi-word state names
        if re.search(r'\b' + re.escape(state) + r'\b', text):
            found_states.add(state)
    return list(found_states)

def process_notes(df, state_names, stop_words, sector_keywords):
    df['Sector'] = df['Note'].apply(lambda x: determine_sector(x, state_names, stop_words, sector_keywords))
    df['States'] = df['Note'].apply(lambda x: extract_states_from_text(x, state_names))
    df = df[['Sector', 'States', 'Domain', 'Note']]
    return df

def main():
    stop_words = load_stopwords()
    state_names = load_state_names('states_data.json')
    df = pd.read_csv('3_govt_urls_state_only.csv')

    bigrams = get_most_common_ngrams(df['Note'], state_names, stop_words, 2)
    trigrams = get_most_common_ngrams(df['Note'], state_names, stop_words, 3)
    most_common_ngrams = {**bigrams, **trigrams}

    sector_keywords = create_dynamic_sector_mapping(list(most_common_ngrams.keys()))

    df = process_notes(df, state_names, stop_words, sector_keywords)
    df.to_csv('classified_data.csv', index=False)
    print(df)

if __name__ == '__main__':
    main()
