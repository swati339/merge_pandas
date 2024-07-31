import nltk
import re
nltk.download()

noise_list = ["is", "a", "this", "..."] 
def _remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

print(_remove_noise("this is a sample text"))

def _remove_regex(input_text, regex_pattern):
    # Use re.sub to replace all matches of the regex pattern in the input_text with an empty string
    return re.sub(regex_pattern, '', input_text).strip()

# Define the regex pattern to match hashtags
regex_pattern = "#[\w]*"  

# Test the function
result = _remove_regex("remove this #hashtag from analytics vidhya", regex_pattern)
print(result)

#Lexicon Normalization

