'''
Write a program for pre-processing of a text document such as stop word removal, stemming

Viva Questions
1) what is stemming, lemmatization

'''

# main NLP library
import nltk
# stopwords - and, the, it, a, an
from nltk.corpus import stopwords
# tokenize - converts words to tokens
from nltk.tokenize import word_tokenize
# contains Stemming algo - converts words to base form
from nltk.stem import PorterStemmer
# re - regular expression
import re

# Download necessary NLTK resources (run this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


# Sample text document
text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language.
The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages in a valuable way.
"""

# Function to preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    # remove anything that is not in lowercase or whitespace
    # re.sub(pattern, replacement, string)
    # sub() - substitute
    # r'[^a-z\s]' - regex that is not (^) in lowercase (a-z) or whitespace (\s)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return stemmed_tokens

# Preprocess the sample text
preprocessed_text = preprocess_text(text)

# Print the preprocessed tokens
print("Preprocessed Tokens:", preprocessed_text)


