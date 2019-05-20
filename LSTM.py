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

data = formatting("phase1_movie_reviews-train.csv")

y = pd.get_dummies(data['polarity'])
X_train, X_dev, y_train, y_dev = train_test_split(data['reviewText'], y, test_size = 0.10, random_state=42)

embedding_size = 300 #number of feature weights in embeddings
max_len = 400

embedding = WordEmbedding(num_features = embedding_size)

WordEmbedding.fit(embedding, X_train)
WordEmbedding.size(embedding)

#Save word embedding to dataframe
#train_embeddings = WordEmbedding.to_pd(embedding, X_train)

#Save Save embeddings to file
WordEmbedding.to_file(embedding)

embeddings_index = import_embedding('trained_embedding_word2vec.txt')

#Basic Vectorization of data
#Review data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

def vectorize(data, tokenizer ,max_len):
    sequences = tokenizer.texts_to_sequences(data)
    padding = pad_sequences(sequences, maxlen = max_len)
    
    return padding

X_train = vectorize(X_train, tokenizer , max_len)
X_dev = vectorize(X_dev, tokenizer, max_len)

print('Found %s unique tokens.' % len(word_index))
print('Shape of train tensor', X_train.shape)
print('Shape of dev tensor', X_dev.shape)


# ## 3. Create word vectors with the loaded word2vec model
embedding_matrix, num_words = embedding_matrix_word2vec(word_index, embedding_size, embeddings_index)


# ### Check train/dev sets
print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_dev.shape)
print('Shape of y_test:', y_dev.shape)


# ## 5. Define model
from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNLSTM, GRU, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

# Define Model
model = Sequential()
model.add(Embedding(num_words, 
                    embedding_size,
                    input_length = max_len,
                     dropout=0.2))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(2, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size = 256, epochs = 4, validation_data = (X_dev, y_dev), verbose = 1)

movie_df = pd.read_csv("phase1_movie_reviews-test-hidden.csv")
movie_labels = pd.read_csv("true_labels/true_movie_labels.txt", header=None)
movies_test = pd.concat([movie_df, movie_labels], axis=1).drop('polarity', axis=1).rename(columns={0: "polarity"})

game_df = pd.read_csv("phase1_video_games-test-hidden.csv")
game_labels = pd.read_csv("true_labels/true_game_labels.txt", header=None)
games_test = pd.concat([game_df, game_labels], axis=1).drop('polarity', axis=1).rename(columns={0: "polarity"})

from keras.preprocessing.text import text_to_word_sequence

movies_test['reviewText'] = movies_test['reviewText'].astype(str)
movies_test['reviewText'] = movies_test['reviewText'].apply(text_to_word_sequence)

games_test['reviewText'] = games_test['reviewText'].astype(str)
games_test['reviewText'] = games_test['reviewText'].apply(text_to_word_sequence)

movies_X_test = movies_test['reviewText']
movies_y_test = movies_test[['polarity']]

games_X_test = games_test['reviewText']
games_y_test = games_test[['polarity']]

movies_X_test = vectorize(movies_X_test, tokenizer , max_len)
games_X_test = vectorize(games_X_test, tokenizer , max_len)

movies_pred = model.predict_classes(movies_X_test)
games_pred = model.predict_classes(games_X_test)

movies_y_test["polarity"] = movies_y_test["polarity"].str.replace('positive', '1')
movies_y_test["polarity"] = movies_y_test["polarity"].str.replace('negative', '0')
movies_y_test["polarity"] = movies_y_test["polarity"].astype('int64')

games_y_test["polarity"] = games_y_test["polarity"].str.replace('positive', '1')
games_y_test["polarity"] = games_y_test["polarity"].str.replace('negative', '0')
games_y_test["polarity"] = games_y_test["polarity"].astype('int64')

from sklearn.metrics import accuracy_score

print(accuracy_score(movies_y_test, movies_pred))
print(accuracy_score(games_y_test, games_pred))