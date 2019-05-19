from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from Preprocessing1.data_format import formatting
import pandas as pd

#create an object of class PorterStemmer
porter = PorterStemmer()
lancaster=LancasterStemmer()

data = formatting('phase1_movie_reviews-train.csv', prune = True)

def lemmatization(sentence):
    for word in range(len(sentence)):
        sentence[word] = lancaster.stem(str(sentence[word]))
    return sentence
    

data['reviewText'].apply(lemmatization)

print("Lemma stemming complete")