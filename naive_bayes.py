import Preprocessing.To_One_Hot as one_hot
from Preprocessing.data_format import formatting
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

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
