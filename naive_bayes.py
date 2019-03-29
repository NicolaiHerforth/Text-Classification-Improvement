import Preprocessing.To_One_Hot as one_hot
from Preprocessing.data_format import formatting
from sklearn.naive_bayes import MultinomialNB
<<<<<<< HEAD
=======
<<<<<<< HEAD
from sklearn import metrics

data = formatting('phase1_movie_reviews-train.csv', prune = True)
# shuffle the dataset to make sure it's not ordered data
data = data.sample(frac=1).reset_index(drop=True)
# if you want to use only a percentage of the dataset, change below variable to whatever percentage you want to use
data = data[:round(len(data)*.1)]
train_data = data[:round(len(data)*.8)]
val_data = data[round(len(data)*.8):]

print('Converting data to dense OneHot encoding')
corpus = one_hot.get_corpus(train_data)

train_data = one_hot.file_to_one_hot(train_data, corpus)
val_data = one_hot.file_to_one_hot(val_data, corpus, test = True)

print('Converted! ')


#X_train = (training.drop(['polarity'], axis=1)).values
X_train = train_data['reviewText'].values
y_train = train_data['polarity'].values
X_val = val_data['reviewText'].values
y_val = (val_data['polarity']).values

clf = MultinomialNB()

clf.fit(list(X_train), y_train)

y_pred_class = clf.predict(list(X_val))

print('Validation Accuracy')
print('  ',metrics.accuracy_score(y_val, y_pred_class))
=======
>>>>>>> e6e7381676ac8ca40a81d8c98ab312b545235232


data = formatting('phase1_movie_reviews-train.csv', prune = True)
print(data)
print('\t Converting data to dense OneHot encoding')
corpus = one_hot.get_corpus(data)

print(len(corpus))

train_data = one_hot.file_to_one_hot(data, corpus)

test_data = one_hot.file_to_one_hot(data, corpus, test = True)

print('\t Converted! ')
training = oh_data[:round(len(oh_data)*.8)]
validation = oh_data[round(len(oh_data)*.8):]

print(training.head(10))
#X_train = (training.drop(['polarity'], axis=1)).values
X_train = training['reviewText'].values
y_train = training['polarity'].values
#X_val = (validation.drop(['polarity'], axis=1)).values
y_val = (validation['polarity']).values

clf = MultinomialNB()

clf.fit(X_train, y_train)

<<<<<<< HEAD
#print(clf.predict(X_val[0:5]))
=======
#print(clf.predict(X_val[0:5]))
>>>>>>> 7de578f132c02868d61ef312c94825d084d5aff6
>>>>>>> e6e7381676ac8ca40a81d8c98ab312b545235232
