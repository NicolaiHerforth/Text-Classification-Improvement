import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from collections import Counter

data = pd.read_csv('../phase1_movie_reviews-train.csv')
data["summary"] = data["summary"].astype(str)
data['reviewText'] = data['reviewText'].astype(str)
data['summary'] = data['summary'].apply(text_to_word_sequence).astype(set)
data['reviewText'] = data['reviewText'].apply(text_to_word_sequence).astype(set)

def file_to_one_hot(data):
    corpus = set()
    data['summary'].apply(corpus.update)
    data['reviewText'].apply(corpus.update)

    # Below creates dixtionary of all words in corpus with values sorted by the frequency of the word.
    # For example the most occuring word in corpus will be called 0, second most 1 and so on..
    words_to_indices = dict()
    for i, word in enumerate(sorted(corpus)):
        words_to_indices[word] = i

    def convert_list_to_ints(l):
        for i in range(len(l)):
            l[i] = words_to_indices[l[i]]
        return l
    print('part3')

    data['summary'] = data['summary'].apply(convert_list_to_ints)
    data['reviewText'] = data['reviewText'].apply(convert_list_to_ints)
    print('part4')
    return data
