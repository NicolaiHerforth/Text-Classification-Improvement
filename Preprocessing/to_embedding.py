import pandas as pd
import numpy as np
import gensim 
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence

class WordEmbedding:
    def __init__(self, num_features = 100, min_word_count = 5, num_workers = 4, window = 5, sg=0):
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.num_workers = num_workers
        self.window = window
        self.sg = sg
        self.model = None
        
    def fit(self, data):
        self.model = gensim.models.Word2Vec(data, 
                                   min_count = self.min_word_count,
                                   size = self.num_features, 
                                   window = self.window, 
                                   workers = self.num_workers)
        return self.model
    
    def size(self):
        print("Total number of words in the vocabulary: ", self.model.wv.syn0.shape)
        
    def _average_word_vectors(self, words, model, vocabulary, num_features):

        feature_vector = np.zeros((num_features,), dtype = "float64")
        n_words = 0.

        for word in words:
            if word in vocabulary: 
                n_words = n_words + 1.
                feature_vector = np.add(feature_vector, model[word])

        if n_words:
            feature_vector = np.divide(feature_vector, n_words)

        return feature_vector

   
    def _averaged_word_vectorizer(self, corpus, model, num_features):
        vocabulary = set(model.wv.index2word)
        features = [self._average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
        return np.array(features)
    
    def to_pd(self, data):
        feature_matrix = self._averaged_word_vectorizer(data, self.model, self.num_features)
        return pd.DataFrame(feature_matrix)
    
    def to_file(self):
        self.model.wv.save_word2vec_format('trained_embedding_word2vec.txt', binary = False)