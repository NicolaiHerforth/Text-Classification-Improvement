import Preprocessing.To_One_Hot as one_hot
from Preprocessing.data_format import formatting
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import collections

data = formatting('phase1_movie_reviews-train.csv', prune = True)
data = data.sample(frac=1).reset_index(drop=True)
data = data[:round(len(data)*.5)]



training = data[:round(len(data)*.8)]
validation = data[round(len(data)*.8):]

y_train = training['polarity'].values

y_val = validation['polarity'].values

counter=collections.Counter(y_train)
print(counter)
most_common = counter.most_common(1)
most_common = most_common[0][0]
print(most_common)

y_base_train = ([most_common]*len(y_train))
y_base_val = ([most_common]*len(y_val))

print('Len y_train : ' + str(len(y_train)))
print('Len y_base : ' + str(len(y_base_train)))

print("Baseline training accuracy")
print("   ", metrics.accuracy_score(y_train, y_base_train))

print("Baseline validation accuracy")
print("   ", metrics.accuracy_score(y_val, y_base_val))




