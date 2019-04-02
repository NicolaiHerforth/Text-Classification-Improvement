import Preprocessing.To_One_Hot as one_hot
from Preprocessing.data_format import formatting
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np

data = formatting('phase1_movie_reviews-train.csv', prune = True)
data = data.sample(frac=1, random_state=1).reset_index(drop=True)

train_data = data[:round(len(data)*.8)]
val_data = data[round(len(data)*.8):]

print('\t Converting data to dense OneHot encoding')
corpus = one_hot.get_corpus(train_data)
max_len = (data['reviewText'].map(len)).max()
padding = 300

train_data = one_hot.file_to_one_hot(train_data, corpus, pad_size= padding)
val_data = one_hot.file_to_one_hot(val_data, corpus, test = True, pad_size= padding)

print('\t Converted! ')

training = train_data
validation = val_data

X_train = training['reviewText'].values
y_train = training['polarity'].values

X_val = validation['reviewText'].values
y_val = (validation['polarity']).values

clf = MultinomialNB()
clf.fit(list(X_train), y_train)

y_train_pred = clf.predict(list(X_train))
y_val_pred = clf.predict(list(X_val))

print('Training Accuracy')
print('  ',metrics.accuracy_score(y_train, y_train_pred))

print('Validation Accuracy')
print('  ',metrics.accuracy_score(y_val, y_val_pred))


movie_data = formatting('phase1_movie_reviews-test-hidden.csv', prune = True)
game_data = formatting('phase1_video_games-test-hidden.csv', prune = True)

movie_data = one_hot.file_to_one_hot(movie_data, corpus, pad_size= padding)
game_data = one_hot.file_to_one_hot(game_data, corpus, test = True, pad_size= padding)

X_movie = movie_data['reviewText'].values
y_movie = movie_data['polarity'].values

X_game = game_data['reviewText'].values
y_game = game_data['polarity'].values

movie_pred = clf.predict(list(X_movie))
game_pred = clf.predict(list(X_game))
print('Exporting hidden game and movie predictions')
np.savetxt("movie_hidden.csv", movie_pred, delimiter=",")
np.savetxt("game_hidden.csv", game_pred, delimiter=",")
print('Exported!')
