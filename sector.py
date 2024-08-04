import pandas as pd
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json

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

def preprocess_text(text, stop_words):
    # Remove everything after '--' if present
    if '--' in text:
        text = text.split('--')[0]
    text = text.strip()  # Remove leading and trailing spaces
    
    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    return words

def generate_ngrams(tokens, n):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(gram) for gram in ngrams]

def get_most_common_ngrams(texts, stop_words, n, min_freq=2):
    all_ngrams = []
    for text in texts:
        tokens = preprocess_text(text, stop_words)
        ngrams = generate_ngrams(tokens, n)
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

def determine_sector(note, stop_words, sector_keywords):
    tokens = preprocess_text(note, stop_words)
    ngrams = generate_ngrams(tokens, 2) + generate_ngrams(tokens, 3)
    for ngram in ngrams:
        for sector, keywords in sector_keywords.items():
            if ngram in keywords:
                return sector
    return 'Unknown'


def extract_states_from_text(text, state_names):
    tokens = preprocess_text(text, [])
    
    state_names_set = set(state_names)
    
    found_states = set()
    
    # Check all possible phrases
    for start in range(len(tokens)):
        for end in range(start + 1, min(start + 6, len(tokens) + 1)):  
            phrase = ' '.join(tokens[start:end])
            if phrase.lower() in state_names_set:
                found_states.add(phrase.lower())
    
    return list(found_states)



def process_notes(df, state_names, stop_words, sector_keywords):
    df['Sector'] = df['Note'].apply(lambda x: determine_sector(x, stop_words, sector_keywords))
    df['States'] = df['Note'].apply(lambda x: extract_states_from_text(x, state_names))
    df = df[['Sector', 'States', 'Domain', 'Note']]
    return df

def main():
    stop_words = load_stopwords()
    state_names = load_state_names('states_data.json')
    df = pd.read_csv('3_govt_urls_state_only.csv')

    bigrams = get_most_common_ngrams(df['Note'], stop_words, 2)
    trigrams = get_most_common_ngrams(df['Note'], stop_words, 3)
    most_common_ngrams = {**bigrams, **trigrams}

    sector_keywords = create_dynamic_sector_mapping(list(most_common_ngrams.keys()))

    df = process_notes(df, state_names, stop_words, sector_keywords)
    df.to_csv('classified_data.csv', index=False)
    print(df)

if __name__ == '__main__':
    main()
