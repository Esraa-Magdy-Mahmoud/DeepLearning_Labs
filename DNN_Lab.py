%matplotlib inline

# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # Free up memory


print('Data and modules loaded.')
features_count = 784
labels_count = 10

# Set the hidden layer width. You can try different widths for different layers and experiment.
hidden_layer_width = 64

#  Set the features, labels, and keep_prob tensors
features = tf.placeholder(tf.float32,shape = (None, features_count))
labels = tf.placeholder(tf.float32,shape = (None, labels_count))

keep_prob = tf.placeholder(tf.float32)


#  Set the list of weights and biases tensors based on number of layers
weights = [
     tf.Variable(tf.truncated_normal([features_count, hidden_layer_width], mean = 0.0, stddev = 0.01)),
     tf.Variable(tf.truncated_normal([hidden_layer_width, hidden_layer_width], mean = 0.0, stddev = 0.01)),
     tf.Variable(tf.truncated_normal([hidden_layer_width, labels_count], mean = 0.0, stddev = 0.01))
]

biases = [
    tf.Variable(tf.zeros([hidden_layer_width])),
    tf.Variable(tf.zeros([hidden_layer_width])),
    tf.Variable(tf.zeros([labels_count]))]


### DON'T MODIFY ANYTHING BELOW ###
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert all(isinstance(weight, Variable) for weight in weights), 'weights must be a TensorFlow variable'
assert all(isinstance(bias, Variable) for bias in biases), 'biases must be a TensorFlow variable'

assert features._shape == None or (\
    features._shape.dims[0].value is None and\
    features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels._shape  == None or (\
    labels._shape.dims[0].value is None and\
    labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'
# TODO: Find the best parameters for each configuration
epochs = 10
batch_size = 64
learning_rate = 0.01
keep_probability = 0.5


### DON'T MODIFY ANYTHING BELOW ###
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels, keep_prob: keep_probability})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict={features: train_features, 
                                                                     labels: train_labels, keep_prob: keep_probability})
                validation_accuracy = session.run(accuracy, feed_dict={features: valid_features, 
                                                                     labels: valid_labels, keep_prob: 1.0})

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict={features: valid_features, 
                                                                     labels: valid_labels, keep_prob: 1.0})

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))
#  Set the epochs, batch_size, and learning_rate with the best parameters from problem 4
epochs = 10
batch_size = 64
learning_rate = 0.01



### DON'T MODIFY ANYTHING BELOW ###
# The accuracy measured against the test set
test_accuracy = 0.0

with tf.Session() as session:
    
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer
            _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels, keep_prob: 1.0})

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict={features: test_features, 
                                                                     labels: test_labels, keep_prob: 1.0})

print('Nice Job! Test Accuracy is {}'.format(test_accuracy))
