import Preprocessing.To_One_Hot as one_hot
from Preprocessing.data_format import formatting
from sklearn.naive_bayes import MultinomialNB


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

#print(clf.predict(X_val[0:5]))