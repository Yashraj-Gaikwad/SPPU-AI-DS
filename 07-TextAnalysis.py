'''
Viva Questions:-

document - text
tokenize - convert text to tokens
pos tags - part of speech tagging
stopwords - is, and, or, i
stemmer - reduce words to base form
lemmatize - reduce words to base forms
tf - count of word in a document
idf - count of word in a group of documents


'''

# nltk = Natural Language Toolkit

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


# Sample document
document = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."

# Tokenization
tokens = word_tokenize(document)

# POS Tagging
pos_tags = pos_tag(tokens)

# Stop words removal
stop_words = set(stopwords.words('english'))
# list comprehension
# stores tokens except stop words
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Calculate Term Frequency (TF)
tf_vectorizer = TfidfVectorizer(use_idf=False)  # for TF
tf_matrix = tf_vectorizer.fit_transform([document]) # sparse matrix
tf_features = tf_vectorizer.get_feature_names_out()

# Calculate Inverse Document Frequency (IDF)
idf_vectorizer = TfidfVectorizer(use_idf=True)  # fro idf
idf_matrix = idf_vectorizer.fit_transform([document])
idf_features = idf_vectorizer.get_feature_names_out()

# Print results
print("Original Document:\n", document)
print("\nTokens:\n", tokens)
print("\nPOS Tags:\n", pos_tags)
print("\nFiltered Tokens (Stop words removal):\n", filtered_tokens)
print("\nStemmed Tokens:\n", stemmed_tokens)
print("\nLemmatized Tokens:\n", lemmatized_tokens)

print("\nTerm Frequency (TF):\n", dict(zip(tf_features, tf_matrix.toarray()[0])))
print("\nInverse Document Frequency (IDF):\n", dict(zip(idf_features, idf_matrix.toarray()[0])))


