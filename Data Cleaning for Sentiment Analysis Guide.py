# Import required libraries for pre-processe text:

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def clean_data(text):

    # Remove URLs from the text
    text = re.sub(r'http\S+', '', text)

    # Remove all non-alphanumeric form the text
    text = re.sub('[^a-zA-z]', ' ', text)

    # Convert text into lower case
    text = str(text).lower()

    # Transform text to tokens
    tokens = word_tokenize(text)

    # Remove stop words from tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]

    # Perform lemmatization
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(token) for token in tokens]

    return tokens

    
# Load dataframe
amazon_review = pd.read_csv("reviews.csv")

amazon_review["cleaned_data"] = amazon_review["Text"].apply(clean_data)
print(amazon_review.head(10))
