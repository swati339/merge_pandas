import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the CSV file into a DataFrame
df = pd.read_csv('3_govt_urls_state_only.csv', encoding='ISO-8859-1', header=None)
num_columns = df.shape[1]
print(num_columns)

df.columns = ['Domain', 'Federal Agency', 'Level of Government', 'Location', 'Status', 'Note', 'Link', 'Date Added']



print('Column names:')
print(df.columns.tolist())

# Define a dictionary mapping sectors to keywords
sector_ngrams = {
    'Travel and Tourism': [
        'travel and tourism', 'tourism site', 'tourist', 'vacation', 'hotel', 'resort',
        'tourist attraction', 'tourism department', 'travel agency', 'travel guide', 'tourism board'
    ],
    'Education': [
        'department of education', 'school', 'university', 'education', 'college', 'learning', 'extension service',
        'academic institution', 'online learning', 'educational program', 'student', 'teacher training'
    ],
    'Food': [
        'food service', 'restaurant', 'cafe', 'cuisine', 'dining', 'eatery',
        'food safety', 'food and beverage', 'catering service', 'gourmet', 'culinary arts'
    ],
    'Public Health': [
        'public health', 'department of public health', 'medical', 'hospital', 
        'healthcare', 'clinic', 'epidemic', 'vaccination', 'health department', 'mental health services'
    ],
    'Government': [
        'state government', 'local government', 'council', 'authority', 'commission', 'department',
        'municipal government', 'federal agency', 'government policy', 'public administration', 'regulatory body'
    ]
}


# Function to extract n-grams from text
def extract_ngrams(text, n=3):
    vectorizer = CountVectorizer(ngram_range=(1, n))
    analyzer = vectorizer.build_analyzer()
    return analyzer(text)

# Function to determine the sector based on the Note content
def determine_sector(note):
    if pd.isna(note):
        return 'Unknown'  # Handle missing Note values
    ngrams = extract_ngrams(note.lower())  # Extract n-grams from note
    for sector, ngram_list in sector_ngrams.items():
        if any(ngram in ngrams for ngram in ngram_list):
            return sector
    return 'Other'  # Return 'Other' if no n-grams match

# Apply the function to the Note column to create a new Sector column
df['Sector'] = df['Note'].apply(determine_sector)

# Display the DataFrame with the new Sector column
print(df[['Domain', 'Note', 'Sector']])
