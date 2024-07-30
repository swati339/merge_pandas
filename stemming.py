import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import nltk

# Download stopwords if not already done
nltk.download('stopwords')
# Download wordnet if not already done
nltk.download('wordnet')


# Example text
text = "I am learning python and enjoying it. It is very important for the data scientists."

# Remove non-alphanumeric characters and convert to lowercase
text_cleaned = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

words = text_cleaned.split()

# Set of English stopwords
stop_words = set(stopwords.words('english'))

# Filter out the stopwords from the list of words
filtered_words = [word for word in words if word not in stop_words]

# Initialize the Porter Stemmer
porter = PorterStemmer()

stemmed = [porter.stem(w) for w in filtered_words]

# Print the stemmed words
print(stemmed)

#Lemmatization
lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
print(lemmed)
