import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [7, 4]

import csv
my_reader = csv.reader(open('phase1_movie_reviews-train.csv'))
positive = 0
negative = 0
for record in my_reader:
    if record[0] == 'positive':
        positive += 1
    elif record[0] == 'negative':
        negative += 1

print('positive : ' + str(positive))
print('neagtive : ' + str(negative))


x = np.arange(2)
acc = [44973,45027]





fig, ax = plt.subplots()

plt.bar(x, acc,width=0.5,align='center',linewidth=2)
plt.xticks(x, ('Positive','Negative'))
plt.ylabel('Polarity Count', fontsize=16)
plt.hlines(45000,-0.25,1.25 ,colors='r', linestyles='dashed', label='')
plt.savefig('polarity count.png')
plt.show()