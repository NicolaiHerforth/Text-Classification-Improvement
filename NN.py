
# coding: utf-8

# In[1]:


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


# In[2]:


data = formatting("phase1_movie_reviews-train.csv")

y = pd.get_dummies(data['polarity'])
X_train, X_dev, y_train, y_dev = train_test_split(data['reviewText'], y, test_size = 0.10, random_state=42)

embedding_size = 300 #number of feature weights in embeddings
max_len = 400


# In[3]:


embedding = WordEmbedding(num_features = embedding_size)

WordEmbedding.fit(embedding, X_train)
WordEmbedding.size(embedding)


# In[4]:


#Save word embedding to dataframe
#train_embeddings = WordEmbedding.to_pd(embedding, X_train)

#Save Save embeddings to file
WordEmbedding.to_file(embedding)


# In[5]:


embeddings_index = import_embedding('trained_embedding_word2vec.txt')


# ## 2. Vectorize text data

# In[6]:


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

# In[7]:


embedding_matrix, num_words = embedding_matrix_word2vec(word_index, embedding_size, embeddings_index)


# ### Check train/dev sets

# In[8]:


print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_dev.shape)
print('Shape of y_test:', y_dev.shape)


# ## 5. Define model

# In[9]:


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

# Define Model
model = Sequential()
model.add(Embedding(num_words, 
                    embedding_size,
                    input_length = max_len,
                     dropout=0.2))
model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(2, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[12]:


history = model.fit(X_train, y_train, batch_size = 256, epochs = 4, validation_data = (X_dev, y_dev), verbose = 1)


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# In[ ]:


loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Development Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_dev, y_dev, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

