import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from collections import Counter

def formatting(path, prune = False):
    data = pd.read_csv(path)
    data["summary"] = data["summary"].astype(str)
    data['reviewText'] = data['reviewText'].astype(str)
    data['summary'] = data['summary'].apply(text_to_word_sequence)
    data['reviewText'] = data['reviewText'].apply(text_to_word_sequence)

    def get_words_to_prune(d):
        w_counts = Counter()
        d['summary'].apply(w_counts.update)
        d['reviewText'].apply(w_counts.update)
        return set([word for word in w_counts if w_counts[word] < 6])
        

    def remove_pruned(words):
        return ' '.join([word for word in words if word not in blacklist])
        
    if prune:
        blacklist = get_words_to_prune(data)
        data['summary'] = data['summary'].apply(remove_pruned)
        data['reviewText'] = data['reviewText'].apply(remove_pruned)
        
    return data


