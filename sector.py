import pandas as pd
import re
from collections import Counter
from nltk.tokenize import word_tokenize

file_path = '3_govt_urls_state_only.csv'
df = pd.read_csv(file_path)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to tokenize and count word frequencies
def get_most_common_words(texts):
    all_words = []
    for text in texts:
        processed_text = preprocess_text(text)
        words = word_tokenize(processed_text)
        all_words.extend(words)
    return Counter(all_words).most_common()

# Extract most common words from the 'Note' column
most_common_words = get_most_common_words(df['Note'])

# Create a list of the most common words
most_common_words_list = [word for word, _ in most_common_words]

# Function to create a dynamic sector mapping
def create_dynamic_sector_mapping(common_words):
    sector_keywords = {}
    
    #dynamically determine sector categories
    for word in common_words:
        sector = word.lower()  # Treat each common word as a potential sector 
        if sector not in sector_keywords:
            sector_keywords[sector] = []
        sector_keywords[sector].append(word)
    
    return sector_keywords

# Create dynamic sector keywords mapping
sector_keywords = create_dynamic_sector_mapping(most_common_words_list)

# Function to determine sector for a given note
def determine_sector(note):
    processed_note = preprocess_text(note)
    words = word_tokenize(processed_note)
    for word in words:
        for sector, keywords in sector_keywords.items():
            if word in keywords:
                return sector
    return 'Unknown'

# Apply the determine_sector function to the 'Note' column
df['Sector'] = df['Note'].apply(determine_sector)

# Save the result to a new CSV file
df[['Domain', 'Note', 'Sector']].to_csv('classified_data.csv', index=False)

# Display the result
print(df[['Domain', 'Note', 'Sector']])
