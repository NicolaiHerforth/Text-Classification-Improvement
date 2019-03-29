import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from collections import Counter
from tqdm import tqdm

def get_corpus(train_data):
    corpus = set()
    train_data['summary'].apply(corpus.update)
    train_data['reviewText'].apply(corpus.update)
    return corpus

def file_to_one_hot(data, corpus, test = False):

    # Below creates dixtionary of all words in corpus with values sorted by the frequency of the word.
    # For example the most occuring word in corpus will be called 0, second most 1 and so on..
    words_to_indices = dict()

    for i, word in enumerate(sorted(corpus)):

        words_to_indices[word] = i + 1

    max_list_length = max([len(max(data['summary'].values, key=len)),len(max(data['reviewText'].values, key=len))])

    def convert_list_to_ints(l):
        for i in range(len(l)):
            l[i] = words_to_indices[l[i]]
        l = l + [len(corpus)] * (max_list_length - len(l))

        return l

    def convert_list_to_ints_for_test(l):
        for i in range(len(l)):
            if l[i] not in corpus:
                l[i] = len(corpus) + 1
            else:
                l[i] = words_to_indices[l[i]]

        l = l + [0] * (max_list_length - len(l))
        return l
    tqdm.pandas()
    if test:

        data['summary'] = data['summary'].progress_apply(convert_list_to_ints_for_test)
        data['reviewText'] = data['reviewText'].progress_apply(convert_list_to_ints_for_test)
    else:
        data['summary'] = data['summary'].progress_apply(convert_list_to_ints)
        data['reviewText'] = data['reviewText'].progress_apply(convert_list_to_ints)

    return data
