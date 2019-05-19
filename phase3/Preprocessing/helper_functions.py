import os
import numpy as np

def import_embedding(file_name):
    embeddings_index = {}
    f = open(os.path.join('', file_name), encoding = "utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close
    
    return embeddings_index
    
def embedding_matrix_word2vec(word_index, embedding_size, embeddings_index):
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_size))
    
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, num_words

def vectorize(data, tokenizer ,max_len):
    sequences = tokenizer.texts_to_sequences(data)
    padding = pad_sequences(sequences, maxlen = max_len)
    
    return padding