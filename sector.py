import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def load_stopwords():
    """Load and return a set of English stopwords."""
    return set(stopwords.words('english'))

def load_state_names(filename):
    """Load and return a set of state names from a JSON file."""
    with open(filename, 'r') as file:
        states_data = json.load(file)
    return {state_info['State'].lower() for state_info in states_data['states info']}

def preprocess_text(text, stop_words):
    """Preprocess text by tokenizing and removing stopwords."""
    text = text.split('--')[0].strip().lower()
    tokens = word_tokenize(text)
    return [token for token in tokens if token not in stop_words]

def generate_ngrams(tokens, n):
    """Generate n-grams from a list of tokens."""
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def filter_ngrams_with_state_names(ngrams, state_names):
    """Filter out n-grams containing any state names."""
    filtered_ngrams = []
    state_name_tokens = {name for state in state_names for name in state.split()}
    for ngram in ngrams:
        if not any(state_name in ngram.split() for state_name in state_name_tokens):
            filtered_ngrams.append(ngram)
    return filtered_ngrams

def get_most_common_ngrams(texts, stop_words, state_names, n, min_freq=1):
    """Get the most common n-grams from a list of texts, excluding state names."""
    all_ngrams = []
    for text in texts:
        tokens = preprocess_text(text, stop_words)
        ngrams = generate_ngrams(tokens, n)
        filtered_ngrams = filter_ngrams_with_state_names(ngrams, state_names)
        all_ngrams.extend(filtered_ngrams)
    ngram_counts = Counter(all_ngrams)
    return {ngram: freq for ngram, freq in ngram_counts.items() if freq >= min_freq}

def create_dynamic_sector_mapping(common_ngrams):
    """Create a dynamic mapping of sectors based on common n-grams."""
    sector_keywords = {}
    for ngram in common_ngrams:
        sector_keywords.setdefault(ngram.lower(), []).append(ngram)
    return sector_keywords

def determine_sector(note, stop_words, sector_keywords, state_names):
    """Determine the sector for a given note."""
    tokens = preprocess_text(note, stop_words)
    tokens = [token for token in tokens if token not in state_names]
    ngrams = generate_ngrams(tokens, 2) + generate_ngrams(tokens, 3)
    for ngram in ngrams:
        for sector, keywords in sector_keywords.items():
            if ngram in keywords:
                return sector
    return 'Unknown'

def extract_states_from_text(text, state_names):
    """Extract state names mentioned in a given text."""
    tokens = preprocess_text(text, [])
    found_states = set()

    for token in tokens:
        if token.lower() in state_names:
            found_states.add(token.lower())

    bigrams = generate_ngrams(tokens, 2)
    for bigram in bigrams:
        if bigram.lower() in state_names:
            found_states.add(bigram.lower())

    return list(found_states)

def process_notes(df, state_names, stop_words, sector_keywords):
    """Process notes in a DataFrame to determine sectors and states."""
    df['Sector'] = df['Note'].apply(lambda x: determine_sector(x, stop_words, sector_keywords, state_names))
    df['States'] = df['Note'].apply(lambda x: extract_states_from_text(x, state_names))
    return df[['Sector', 'States', 'Domain', 'Note']]

def main():
    """Main function to execute the processing of notes."""
    stop_words = load_stopwords()
    state_names = load_state_names('states_data.json')
    df = pd.read_csv('3_govt_urls_state_only.csv')

    # Generate common n-grams
    bigrams = get_most_common_ngrams(df['Note'], stop_words, state_names, 2)
    trigrams = get_most_common_ngrams(df['Note'], stop_words, state_names, 3)
    most_common_ngrams = {**bigrams, **trigrams}

    # Create sector keywords mapping
    sector_keywords = create_dynamic_sector_mapping(list(most_common_ngrams.keys()))

    # Process notes and save results
    df = process_notes(df, state_names, stop_words, sector_keywords)
    df.to_csv('classified_data_1.csv', index=False)
    print(df)

if __name__ == '__main__':
    main()
