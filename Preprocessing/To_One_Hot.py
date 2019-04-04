from keras.preprocessing.text import text_to_word_sequence
from collections import Counter


def get_corpus(train_data):
    corpus = Counter()
    train_data['reviewText'].apply(corpus.update)
    return corpus


def file_to_one_hot(data, corpus, test = None):
    words_to_indices = dict()
    count = 0
    for word in sorted(corpus):
        if corpus[word] > 50:
            words_to_indices[word] = count
            count += 1


    def convert_list_to_ints_for_test(l):
        ll = len(words_to_indices) * [0]

        for i in l:
            if i not in words_to_indices:
                continue
            else:
                ll[words_to_indices[i]] = 1

        return ll

    data['reviewText'] = data['reviewText'].apply(convert_list_to_ints_for_test)
    return data

