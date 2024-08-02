import pandas as pd
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json

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

# Remove the header row or other unwanted rows if necessary
state_names = [name for name in state_names if name.lower() != 'code']

# Load CSV file
file_path = '3_govt_urls_state_only.csv'
df = pd.read_csv(file_path)

# Function to preprocess text
def preprocess_text(text):
    # Remove everything after '--' if present
    if '--' in text:
        text = text.split('--')[0]
    text = text.strip()  # Remove leading and trailing spaces
    
    # Remove state names
    for state in state_names:
        text = re.sub(r'\b' + re.escape(state) + r'\b', '', text, flags=re.IGNORECASE)
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

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
    # Placeholder for dynamic sector keywords mapping
    sector_keywords = {}
    
    #dynamically determine sector categories
    for word in common_words:
        sector = word.lower()  
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

# Reorder columns to place 'Sector' first
df = df[['Sector', 'Domain', 'Note']]

# Save the result to a new CSV file
df.to_csv('classified_data.csv', index=False)

# Display the result
print(df)
