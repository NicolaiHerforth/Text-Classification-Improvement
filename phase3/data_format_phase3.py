import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
from afinn_sentiment import avg_afinn_sentiment
from afinn import Afinn
from topic_modelling import add_topic_features
import pickle
from nltk.stem import LancasterStemmer





def formatting(path, test = False):

    af = Afinn()
    lancaster=LancasterStemmer()


    def get_words_to_prune(d, k = False):

        stop_words = set(stopwords.words('english'))
        w_counts = Counter()
        d['summary'].apply(w_counts.update)
        d['reviewText'].apply(w_counts.update)

        return set([word for word in w_counts if w_counts[word] < 10]) | stop_words

    def remove_pruned(words):
        return [word for word in words if word not in blacklist]

    def lemmatization(sentence):
        return [lancaster.stem(word) for word in sentence]

    
    data = pd.read_csv(path)
    data = data.fillna('')


    data["summary"] = data["summary"].astype(str)
    data['reviewText'] = data['reviewText'].astype(str)
    data['summary'] = data['summary'].apply(text_to_word_sequence)
    data['reviewText'] = data['reviewText'].apply(text_to_word_sequence)

    data['reviewText'] = data['reviewText'] + data['summary']
    data.drop('summary', axis = 1)

    if test:
        blacklist = pickle.load(open("blacklisted_words", "rb"))
    else:
        blacklist = get_words_to_prune(data)
        pickle.dump(blacklist, open("blacklisted_words", "wb" ))            

    data['reviewText'] = data['reviewText'].apply(remove_pruned)

    data = add_topic_features(data)

    def forward(smth):
        return avg_afinn_sentiment(smth,af)

    data['affin_score'] = data['reviewText'].apply(forward)
    data['reviewText'] = data['reviewText'].apply(lemmatization)
    return data





