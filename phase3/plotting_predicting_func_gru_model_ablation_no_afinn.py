#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from Preprocessing.to_embedding import WordEmbedding
from data_format_phase3 import formatting
from Preprocessing.helper_functions import import_embedding, embedding_matrix_word2vec
from sklearn.model_selection import train_test_split


# In[2]:


import keras

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

keras.backend.tensorflow_backend._get_available_gpus()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[3]:


file_path = open("fomatted_data",'rb')
data = pickle.load(file_path)


# In[4]:


data = data.drop("year", axis=1)

#data = data[:round(len(data)*.2)]
y = pd.get_dummies(data['polarity'])
X_train, X_dev, y_train, y_dev = train_test_split(data, y, test_size = 0.20, random_state=42)

X_train_nlp, X_dev_nlp = X_train['reviewText'], X_dev['reviewText']

X_train_meta, X_dev_meta = X_train.iloc[:,3:], X_dev.iloc[:,3:]
embedding_size = 300 #number of feature weights in embeddings
max_len = 400


# In[5]:


#Basic Vectorization of data
#Review data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_nlp)
word_index = tokenizer.word_index

def vectorize(data, tokenizer ,max_len):
    sequences = tokenizer.texts_to_sequences(data)
    padding = pad_sequences(sequences, maxlen = max_len)
    
    return padding

X_train_nlp = vectorize(X_train_nlp, tokenizer , max_len)
X_dev_nlp = vectorize(X_dev_nlp, tokenizer, max_len)

print('Found %s unique tokens.' % len(word_index))
print('Shape of train tensor', X_train_nlp.shape)
print('Shape of dev tensor', X_dev_nlp.shape)


# ## Game Data

# In[6]:


game_df = pd.read_csv("../phase1_video_games-test-hidden.csv")
game_labels = pd.read_csv("../true_labels/true_game_labels.txt", header=None)
merged = pd.concat([game_df, game_labels], axis=1).drop('polarity', axis=1).rename(columns={0: "polarity"})



# get a list of columns

cols = list(merged)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('polarity')))
cols

# use ix to reorder
merged = merged.ix[:, cols]
merged.to_csv('merged_df.csv', index=False)



from data_format_phase3 import formatting
game_data = formatting("merged_df.csv", test=True)

game_data = game_data.drop('year', axis=1)
game_data = game_data.drop('affin_score', axis=1)

game_y = pd.get_dummies(game_data['polarity'])




"""HVIS DU ÆNDRE HVOR MANGE FEATURES DER ER, SKAL DU ÆNDRE INDEXERING HER:"""
game_train_meta = game_data.iloc[:,3:]


game_train_nlp = game_data['reviewText']

game_train_nlp = vectorize(game_train_nlp, tokenizer , max_len)
game_sets = [game_train_nlp, game_train_meta]


# In[7]:


game_data

#game_data.drop('affin_score', axis=1, inplace=True)




# In[ ]:





# In[ ]:





# In[ ]:





# ## Movie Data

# In[8]:


movie_df = pd.read_csv("../phase1_movie_reviews-test-hidden.csv")
movie_labels = pd.read_csv("../true_labels/true_movie_labels.txt", header=None)
merged = pd.concat([movie_df, movie_labels], axis=1).drop('polarity', axis=1).rename(columns={0: "polarity"})
# get a list of columns
cols = list(merged)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('polarity')))
cols

# use ix to reorder
merged = merged.ix[:, cols]
merged.to_csv('merged_df.csv', index=False)
from data_format_phase3 import formatting
movie_data = formatting("merged_df.csv", test=True)

movie_data = movie_data.drop('year', axis=1)
movie_data = movie_data.drop('affin_score', axis=1)

movie_y = pd.get_dummies(movie_data['polarity'])



"""HVIS DU ÆNDRE HVOR MANGE FEATURES DER ER, SKAL DU ÆNDRE INDEXERING HER:"""
movie_train_meta = movie_data.iloc[:,3:]


movie_train_nlp = movie_data['reviewText']

movie_train_nlp = vectorize(movie_train_nlp, tokenizer , max_len)
movie_sets = [movie_train_nlp, movie_train_meta]






# In[9]:


from keras.models import load_model
# Returns a compiled model identical to the previous one
model = load_model('funct_GRU_model_ablation_no_afinn.h5')


# In[10]:


predicted_game = model.predict(x=game_sets,batch_size=200, verbose=1)


#ARGMAX PREDICTIONS GAME
for i in predicted_game:
    max_idx = np.argmax(i)
    if max_idx == 0:
        i[0] = 1
        i[1] = 0
    else:
        i[1] = 1
        i[0] = 0
        
        


predicted_movie = model.predict(x=movie_sets,batch_size=200, verbose=1)
        
#ARGMAX PREDICTIONS MOVIE
for i in predicted_movie:
    max_idx = np.argmax(i)
    if max_idx == 0:
        i[0] = 1
        i[1] = 0
    else:
        i[1] = 1
        i[0] = 0
        


# In[11]:


game_y = game_y.as_matrix()
print(game_y)
print(predicted_game)

print(type(game_y))
print(type(predicted_game))


# In[12]:


movie_y = movie_y.as_matrix()
print(movie_y)
print(predicted_movie)

print(type(game_y))
print(type(predicted_movie))


# ## Accuracy score Game data: 
# 

# In[13]:


from sklearn.metrics import accuracy_score


# In[14]:


y_pred_game = predicted_game
y_true_game = game_y
accuracy_score(y_true_game, y_pred_game)


# ## Accuracy score movie data: 

# In[15]:


y_pred_movie = predicted_movie
y_true_movie = movie_y
accuracy_score(y_true_movie, y_pred_movie)


# ## Precision, Recall, f1-score for Game Data

# In[16]:


from sklearn.metrics import classification_report


# In[17]:


target_names = ['class 0', 'class 1']
print(classification_report(game_y, predicted_game, target_names=target_names))


# ## Precision, Recall, f1-score for Movie Data

# In[18]:


target_names = ['class 0', 'class 1']
print(classification_report(movie_y, predicted_movie, target_names=target_names))


# In[ ]:




