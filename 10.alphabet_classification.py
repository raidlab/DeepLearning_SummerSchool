import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from functions import *

# Load given dataset
train_data = pickle.load(open('alphabet/train_data.txt', 'rb'))
train_labels = pickle.load(open('alphabet/train_labels.txt', 'rb'))
test_data = pickle.load(open('alphabet/test_data.txt', 'rb'))
test_labels = pickle.load(open('alphabet/test_labels.txt', 'rb'))

nb_train = train_data.shape[0]      # the number of train dataset
nb_test = test_data.shape[0]        # the number of test dataset
nb_features = train_data.shape[1]   # the number of features
nb_classes = train_labels.shape[1]  # the number of classes

# Input(features): 1D intensity array 24 * 24 = 576 : resized features
# Output(classes): A to Z = 26
X = tf.placeholder(tf.float32, [None, nb_features])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# Deep NN: 2 Hidden layer
tf.set_random_seed(777) # reproducibility when initializing layers
keep_prob = tf.placeholder(tf.float32) # train: 0.5 ~ 0.7 / test: 1

nb_layer1 = 300 # Resolution
W1 = tf.get_variable('weight1', shape=[nb_features, nb_layer1], initializer=tf.contrib.layers.xavier_initializer()) # Initialization of Weights
b1 = tf.Variable(tf.random_normal([nb_layer1]), name='bias1') # Initialization of Bias
L1 = tf.nn.relu(tf.matmul(X, W1) + b1) # Activation function
L1 = tf.nn.dropout(L1, keep_prob=keep_prob) # dropout

nb_layer2 = 300 # Resolution
W2 = tf.get_variable('weight2', shape=[nb_layer1, nb_layer2], initializer=tf.contrib.layers.xavier_initializer()) # Initialization of Weights
b2 = tf.Variable(tf.random_normal([nb_layer2]), name='bias2') # initialization of Bias
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2) # Activation function
L2 = tf.nn.dropout(L2, keep_prob=keep_prob) # dropout

W3 = tf.get_variable('weight3', shape=[nb_layer2, nb_classes], initializer=tf.contrib.layers.xavier_initializer()) # Initialization of Weights
b3 = tf.Variable(tf.random_normal([nb_classes]), name='bias3') # Initialization of Bias
hypothesis = tf.matmul(L2, W3) + b3

# Cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)) # Cross entropy for hypothesis with softmax classification
# line 65 is equal to following two lines.
# hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cost) # Gradient descent Optimize

# Calculate accuracy
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)) # If predicted result is equal to classe, TURE is returned.
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # Definition of accuracy

# hyper-parameters
nb_epochs = 500
batch_size = 25
iter_per_epochs = int(nb_train / batch_size)

tic() # training timer start

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epoch_history = []
cost_history = []

# Visulization of avg. cost
plt.ion() # Turn interactive mode on.
fig = plt.figure() # Open Window as fig
sf = fig.add_axes([0.15, 0.1, 0.8, 0.8]) # Define Drawing Area
plt.xlim([0, nb_epochs]) # X Axis limit
plt.ylim([0, 5])

line, = sf.plot(epoch_history, cost_history, 'b-', lw=1)
plt.ylabel('Cost')
plt.xlabel('Epoch')

# Training cycle
for epoch in range(nb_epochs):
    avg_cost = 0

    for i in range(iter_per_epochs):
        batch_xs, batch_ys = next_batch(batch_size, train_data, train_labels) # Training dataset의 무작위 선택
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7}) # Training session
        avg_cost += c / iter_per_epochs

    epoch_history.append(epoch), cost_history.append(avg_cost)

    line.set_xdata(epoch_history)
    line.set_ydata(cost_history)

    plt.draw(), plt.pause(0.00001)

    if (epoch % 10) == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

toc()
print("Learning finished")

# Test the model using test sets
train_accuracy = sess.run(accuracy, feed_dict={X: train_data, Y: train_labels, keep_prob: 1})
test_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_labels, keep_prob: 1})
print("\nTrain Acc: ", train_accuracy, "\nTest Acc: ", test_accuracy)

print("END")
