import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize, RegexpParser

# Example text
sample_text = "The quick brown fox jumps over the lazy dog"

# Tokenize the text and find parts of speech
tagged = pos_tag(word_tokenize(sample_text))

# Define a grammar for chunking
chunker = RegexpParser("""
    NP: {<DT>?<JJ>*<NN>}     # Noun Phrase
    P: {<IN>}                # Preposition
    V: {<V.*>}               # Verb
    PP: {<IN><NP>}           # Prepositional Phrase
    VP: {<V.*><NP|PP>*}      # Verb Phrase
""")

# Parse the sentence to extract chunks
tree = chunker.parse(tagged)

# Print the chunks
for subtree in tree:
    if isinstance(subtree, nltk.Tree):
        label = subtree.label()
        leaves = " ".join([word for word, pos in subtree.leaves()])
        print(f"{label}: {leaves}")
    else:
        word, pos = subtree
        print(f"Other: {word} ({pos})")
