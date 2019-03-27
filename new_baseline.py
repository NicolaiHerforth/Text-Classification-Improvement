import Preprocessing.To_One_Hot as one_hot
from Preprocessing.data_format import formatting
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import collections

data = formatting('phase1_movie_reviews-train.csv', prune = True)
data = data.sample(frac=1).reset_index(drop=True)
data = data[:round(len(data)*.005)]


#print(data.head())

print('\t Converting data to dense OneHot encoding')
corpus = one_hot.get_corpus(data)

train_data = one_hot.file_to_one_hot(data, corpus)

#print(train_data.head())

test_data = one_hot.file_to_one_hot(data, corpus, test = True)

print('\t Converted! ')
training = train_data[:round(len(train_data)*.8)]
validation = train_data[round(len(train_data)*.8):]



X_train = training['reviewText'].values
y_train = training['polarity'].values
X_val = validation['reviewText'].values
y_val = validation['polarity'].values

counter=collections.Counter(y_train)
print(counter)
most_common = counter.most_common(1)
most_common = most_common[0][0]
print(most_common)

#y_base = (*len(y_train))

print('Len y_train : ' + str(len(y_train)))
#print('Len y_base : ' + str(len(y_base)))

#print(collections.Counter(y_base))

