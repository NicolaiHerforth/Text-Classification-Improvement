import Preprocessing.To_One_Hot as one_hot
from Preprocessing.data_format import formatting
from sklearn.naive_bayes import MultinomialNB
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
