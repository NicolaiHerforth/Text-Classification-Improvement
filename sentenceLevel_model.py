import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from Preprocessing.to_embedding import WordEmbedding
from Preprocessing.data_format import formatting
from Preprocessing.helper_functions import import_embedding, embedding_matrix_word2vec
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence
import keras

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

keras.backend.tensorflow_backend._get_available_gpus()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

data = pd.read_csv('phase2_baby_all.csv')

data["sentence"] = data["sentence"].astype(str)
data["sentence"] = data["sentence"].apply(text_to_word_sequence)

embedding_size = 300 #number of feature weights in embeddings
max_len = 400

avg_len = sum(data["sentence"].str.len())/len(data["sentence"])
max_len = data["sentence"].str.len()

embedding = WordEmbedding(num_features = embedding_size)
WordEmbedding.fit(embedding, data["sentence"])
WordEmbedding.size(embedding)

#Save word embedding to dataframe
#train_embeddings = WordEmbedding.to_pd(embedding, X_train)

#Save Save embeddings to file
WordEmbedding.to_file(embedding)

embeddings_index = import_embedding('trained_embedding_word2vec.txt')


def vectorize(data, tokenizer ,max_len):
    sequences = tokenizer.texts_to_sequences(data)
    padding = pad_sequences(sequences, maxlen = max_len)
    
    return padding


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNGRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from sklearn.model_selection import StratifiedKFold


cvscores = []
for group_id in range(1,9):
    train = data.loc[data['group_id'] != group_id]
    X_train = train['sentence']
    y_train = train['sentiment']
    
    test = data.loc[data['group_id'] == group_id]
    X_test = test['sentence']
    y_test = test['sentiment']
    
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    
    embedding_matrix, num_words = embedding_matrix_word2vec(word_index, embedding_size, embeddings_index)
    
    X_train = vectorize(X_train, tokenizer , max_len)
    X_test = vectorize(X_test, tokenizer, max_len)

    # Define Model
    model = Sequential()
    model.add(Embedding(num_words, 
                        embedding_size,
                        input_length = max_len,
                         dropout=0.2))
    model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model.fit(X_train, y_train, batch_size = 64, epochs = 3, verbose = 1)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1])
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print(cvscores)
