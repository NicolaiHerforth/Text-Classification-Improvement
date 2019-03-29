
# coding: utf-8

# In[7]:


import Preprocessing.To_One_Hot as one_hot
from Preprocessing.data_format import formatting
import numpy as np
import tensorflow as tf
import pandas as pd
from string import punctuation


# In[8]:


data = pd.read_csv('phase1_movie_reviews-train.csv')
# shuffle the dataset to make sure it's not ordered data
#data = data.sample(frac=1).reset_index(drop=True)
# if you want to use only a percentage of the dataset, change below variable to whatever percentage you want to use
data = data[:round(len(data)*.1)]


# In[9]:


data['reviewText'][0]


# In[10]:


reviews = data['reviewText']
reviews = reviews.str.cat(sep='\n')

labels_org = data['polarity']
labels_org = labels_org.str.cat(sep='\n')


# In[11]:


reviews[:2000]


# In[12]:


all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()


# In[13]:


all_text[:200]


# In[14]:


words[:10]


# In[15]:


# Create your dictionary that maps vocab words to integers here
vocab = set(words)
vocab_to_int = {w: i for i, w in enumerate(vocab, 1)}
print(len(vocab_to_int))

# Convert the reviews to integers, same shape as reviews list, but with integers
reviews_ints = []

for r in reviews:
    ri = [vocab_to_int.get(w) for w in r if vocab_to_int.get(w) is not None]
    reviews_ints.append(ri)

reviews_ints[:10]

from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

reviews_ints = []

for each in reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split()])


# In[16]:


print(len(reviews_ints))


# In[17]:


# Convert labels to 1s and 0s for 'positive' and 'negative'
# print(labels_org)
labels = np.array([1 if l == "positive" else 0 for l in labels_org.split()])
# print(labels)
print(len(labels))


# In[18]:


from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# In[19]:


# Filter out that review with 0 length
reviews_ints = [r[0:200] for r in reviews_ints if len(r) > 0]


# In[20]:


from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# In[21]:


seq_len = 200
features = np.zeros((len(reviews_ints), seq_len), dtype=int)
# print(features[:10,:100])
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
features[:10,:11]


# In[22]:


print(len(features))
print(type(features))
print(features[500])
print(len(features[500]))
print(reviews_ints[500])
print(len(reviews_ints[500]))


# In[23]:


split_frac = 0.8

split_index = int(split_frac * len(features))

train_x, val_x = features[:split_index], features[split_index:] 
train_y, val_y = labels[:split_index], labels[split_index:]

split_frac = 0.5
split_index = int(split_frac * len(val_x))

val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("label set: \t\t{}".format(train_y.shape), 
      "\nValidation label set: \t{}".format(val_y.shape),
      "\nTest label set: \t\t{}".format(test_y.shape))


# In[24]:


lstm_size = 256
lstm_layers = 2
batch_size = 1000
learning_rate = 0.01


# In[25]:


n_words = len(vocab_to_int) + 1 # Add 1 for 0 added to vocab

# Create the graph object
tf.reset_default_graph()
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")


# In[26]:


# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300 

with tf.name_scope("Embeddings"):
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)


# In[27]:


def lstm_cell():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    # Add dropout to the cell
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

with tf.name_scope("RNN_layers"):
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


# In[28]:


with tf.name_scope("RNN_forward"):
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


# In[29]:


with tf.name_scope('predictions'):
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    tf.summary.histogram('predictions', predictions)
with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(labels_, predictions)
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

merged = tf.summary.merge_all()


# In[30]:


with tf.name_scope('validation'):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[31]:


def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# In[ ]:


epochs = 10

# with graph.as_default():
#saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #train_writer = tf.summary.FileWriter('./logs/tb/train', sess.graph)
    #test_writer = tf.summary.FileWriter('./logs/tb/test', sess.graph)
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            summary, loss, state, _ = sess.run([merged, cost, final_state, optimizer], feed_dict=feed)
#             loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            #train_writer.add_summary(summary, iteration)
        
            #if iteration%5==0:
            print("Epoch: {}/{}".format(e, epochs),
                  "Iteration: {}".format(iteration),
                  "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
#                     batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    summary, batch_acc, val_state = sess.run([merged, accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
            #test_writer.add_summary(summary, iteration)
            #saver.save(sess, "checkpoints/sentiment_manish.ckpt")
    #saver.save(sess, "checkpoints/sentiment_manish.ckpt")


# In[ ]:




