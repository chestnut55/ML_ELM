import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from elm import GenELMClassifier, BaseELM, ELMClassifier
from random_layer import RBFRandomLayer, GRBFRandomLayer
import data_load as load


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


L1 = 32  # number of convolutions for first layer
L2 = 64  # number of convolutions for second layer
L3 = 1024  # number of neurons for dense layer
learning_date = 1e-4  # learning rate
epochs = 20  # number of times we loop through training data
batch_size = 10  # number of data per batch

train_data, test_data, train_labels, test_labels = load.obesity_data()
features = train_data.shape[1]
classes = 2
sess = tf.InteractiveSession()

xs = tf.placeholder(tf.float32, [None, features])
ys = tf.placeholder(tf.float32, [None, classes])
keep_prob = tf.placeholder(tf.float32)
x_shape = tf.reshape(xs, [-1, 1, features, 1])

# first conv
w_conv1 = weight_variable([5, 5, 1, L1])
b_conv1 = bias_variable([L1])
h_conv1 = tf.nn.relu(conv2d(x_shape, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second conv
w_conv2 = weight_variable([5, 5, L1, L2])
b_conv2 = bias_variable([L2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 117 * L2])

# third dense layer,full connected
w_fc1 = weight_variable([1 * 117 * L2, L3])
b_fc1 = bias_variable([L3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fourth layer, output
w_fc2 = weight_variable([L3, classes])
b_fc2 = bias_variable([classes])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_mean(ys * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_date).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

train_features_cnn = np.zeros((len(train_data), L3), dtype=float)
train_labels_cnn = np.zeros(len(train_data), dtype=int)
test_labels_cnn = np.zeros(len(test_data), dtype=int)


converter = np.array([0, 1])
### start train###
def conv_net_classifier():
    for epoch in range(epochs):
        # print 'epoch: ' + str(epoch)
        for batch in range(len(train_data) // batch_size):
            offset = (batch * batch_size) % len(train_data)
            batch_data = train_data[offset:(offset + batch_size)]
            batch_labels = train_labels[offset:(offset + batch_size)]

            features_batch = h_fc1.eval(feed_dict={xs: batch_data})

            for j in range(batch_size):
                for k in range(L3):
                    train_features_cnn[batch_size * batch + j, k] = features_batch[j, k]
                train_labels_cnn[batch_size * batch + j] = np.sum(np.multiply(converter, batch_labels[j, :]))

    test_features_cnn = h_fc1.eval(feed_dict={xs: test_data})
    for j in range(len(test_data)):
        test_labels_cnn[j] = np.sum(np.multiply(converter, test_labels[j, :]))

    clf = svm.SVC()
    # clf = RandomForestClassifier(
    #     n_estimators=1000, random_state=0).fit(train_features_cnn, train_labels_cnn)
    # srhl_rbf = RBFRandomLayer(n_hidden=50, rbf_width=0.1, random_state=0)
    # clf = GenELMClassifier(hidden_layer=srhl_rbf)
    clf.fit(train_features_cnn, train_labels_cnn)
    accuracy = clf.score(test_features_cnn, test_labels_cnn)
    print "ConvNet classifier accuracy ="+ str(accuracy)

def conv_net(accuracy):
    ### start train###
    for epoch in range(epochs):
        # print 'epoch: ' + str(epoch)
        for batch in range(len(train_data) // batch_size):
            offset = (batch * batch_size) % len(train_data)
            batch_data = train_data[offset:(offset + batch_size)]
            batch_labels = train_labels[offset:(offset + batch_size)]
            train_step.run(feed_dict={xs: batch_data, ys: batch_labels, keep_prob: 0.5})
    ### test###
    accuracy = accuracy.eval(feed_dict={xs: test_data, ys: test_labels, keep_prob: 1.0})
    print "conv_net accuracy= " + str(accuracy)
conv_net(accuracy)
conv_net_classifier()
sess.close()
