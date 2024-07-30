import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Define the stop words
stop_words = set(stopwords.words('english'))

# Sample text
txt = "Natural language processing is an exciting area. Huge budgets have been allocated for this."

# Tokenize the text into sentences
tokenized = sent_tokenize(txt)

# Process each sentence
for i in tokenized:
    # Tokenize the sentence into words
    wordsList = word_tokenize(i)
    # Remove stopwords
    wordsList = [w for w in wordsList if not w.lower() in stop_words]
    # POS tagging for each word
    tagged = nltk.pos_tag(wordsList)
    # Print the tagged words
    print(tagged)
