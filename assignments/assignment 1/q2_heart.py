import numpy as np
import tensorflow as tf
import time
import pandas as pd

# Define paramaters for the model
learning_rate = 0.01
batch_size = 10
n_epochs = 100

# Step 1: Read in data
dat = pd.read_csv('D:\\Dropbox\\study (ms)\\self study\\cs20si\\data\\heart.csv')
dat['famhist'] = (np.where(dat['famhist']=='Present', 1, 0))

# Split into training and test data sets
n = dat.shape[0]
np.random.seed(2017)
n_train = int(n*.7)
n_test = n - n_train
ind = np.random.permutation(n)
train_data = dat.iloc[ind[:n_train],:]
test_data = dat.iloc[ind[n_train:],:]
train_labels = np.array(train_data['chd'])
test_labels = np.array(test_data['chd'])
cols = [col for col in dat.columns if col != 'chd']
train_var = train_data[cols].as_matrix()
test_var = test_data[cols].as_matrix()

# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, [None, 9], name='X_placeholder') 
Y = tf.placeholder(tf.int32, [None], name='Y_placeholder')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.1
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.Variable(tf.random_normal(shape=[9, 1], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1]), name="bias")

# Step 4: build model
outputs = tf.sigmoid(tf.reduce_sum(tf.matmul(X, w), [1]) + b)
predictions = tf.cast(tf.greater_equal(outputs, 0.5), tf.int32)

loss_tmp = -tf.to_float(Y) * tf.log(outputs) - (1.0 - tf.to_float(Y)) * tf.log(1.0 - outputs)
loss = tf.reduce_mean(loss_tmp)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# define own next_batch function for data set
def next_batch(num, data, labels):
	idx = np.arange(0, len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[i] for i in idx]
	labels_shuffle = [labels[i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)
  
  
# Train the model
with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(len(train_var)/batch_size)
  
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0.0
		for _ in range(n_batches):
			X_batch, Y_batch = next_batch(batch_size, train_var, train_labels)
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch}) 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!')

	# test the model
	
	correct_preds = tf.equal(predictions, Y)
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
	
	total_correct_preds = 0
	
	accuracy_test = sess.run([accuracy], feed_dict={X: test_var, Y:test_labels})
	total_correct_preds += accuracy_test[0]
	
	print('Accuracy {0}'.format(total_correct_preds/len(test_labels)))
