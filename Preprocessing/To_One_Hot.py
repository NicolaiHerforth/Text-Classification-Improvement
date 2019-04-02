
from keras.preprocessing.text import text_to_word_sequence
from collections import Counter


def get_corpus(train_data):
    corpus = Counter()
    #train_data['summary'].apply(corpus.update)
    train_data['reviewText'].apply(corpus.update)

    return corpus

    
def file_to_one_hot(data, corpus, pad_size = 111, test = None):
    #corpus = set()
    #data['summary'].apply(corpus.update)
    #data['reviewText'].apply(corpus.update)
    words_to_indices = dict()

    for i, word in enumerate(sorted(corpus)):
        words_to_indices[word] = i


    def convert_list_to_ints_for_test(l):
        ll = len(corpus) * [0]

        for i in l:
            if i not in corpus or corpus[i] < 10:
                continue
            else:
                ll[words_to_indices[i]] = 1

        return ll


    #data['summary'] = data['summary'].apply(convert_list_to_ints_for_test)
    data['reviewText'] = data['reviewText'].apply(convert_list_to_ints_for_test)

    return data

'''

import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from collections import Counter
from tqdm import tqdm

def get_corpus(train_data):
    corpus = set()
    train_data['summary'].apply(corpus.update)
    train_data['reviewText'].apply(corpus.update)
    return corpus

def file_to_one_hot(train, test, corpus, pad_length = 111):

    x_train, x_test = list(train['summary']), list(test['summary'])
    x_train, x_test = list(test['reviewText']), list(test['reviewText'])

    print('hey')
    word_counts = Counter()
    #train['summary'].apply(word_counts.update)
    train['reviewText'].apply(word_counts.update)
    print('wuhu')
    def integerify_by_freq(x_train, x_test):

        sorted_words = list(reversed(sorted(word_counts, key=lambda k: word_counts[k])))
        
        def convert_int(x):
            print(len(x))
            for i, l in enumerate(x):
                if i % 1000 == 0:
                    print(i)
                some_list = []
                for word in l:
                    if word in sorted_words:
                        some_list.append(sorted_words.index(word)+1)
                    else:
                        some_list.append(0)
                x[i]= some_list
            return x
        
        x_train = convert_int(x_train) 
        x_test = convert_int(x_test)

        return x_train, x_test

    train_data, test_data = integerify_by_freq(x_train, x_test)
    train_data = sequence.pad_sequences(train_data, maxlen = pad_length)
    test_data = sequence.pad_sequences(test_data, maxlen = pad_length)

    return train_data, test_data

'''
