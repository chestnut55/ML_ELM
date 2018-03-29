"""
CNN and Classifier model, CNN first then Classifier
CNN - for the futures extractions
Classifier - for output classifications
"""
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
import math


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_net_classifier():
    L1 = 32  # number of convolutions for first layer
    L2 = 64  # number of convolutions for second layer
    L3 = 1024  # number of neurons for dense layer
    learning_date = 1e-3  # learning rate
    epochs = 1  # number of times we loop through training data
    batch_size = 10  # number of data per batch

    train_data, test_data, train_labels, test_labels = load.hmp_hmpii_data()
    features = train_data.shape[1]
    print "features:"+str(features)
    classes = train_labels.shape[1]
    sess = tf.InteractiveSession()

    xs = tf.placeholder(tf.float32, [None, features])
    ys = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)
    x_shape = tf.reshape(xs, [-1, 1, features, 1])

    # first conv
    w_conv1 = weight_variable([3, 3, 1, L1])
    b_conv1 = bias_variable([L1])
    h_conv1 = tf.nn.relu(conv2d(x_shape, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second conv
    w_conv2 = weight_variable([3, 3, L1, L2])
    b_conv2 = bias_variable([L2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    tmp_shape = (int)(math.ceil(features/4.0))
    print tmp_shape
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * tmp_shape * L2])

    # third dense layer,full connected
    w_fc1 = weight_variable([1 * tmp_shape * L2, L3])
    b_fc1 = bias_variable([L3])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fourth layer, output
    w_fc2 = weight_variable([L3, classes])
    b_fc2 = bias_variable([classes])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_date).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_features_cnn = np.zeros((len(train_data), L3), dtype=float)
    train_labels_cnn = np.zeros(len(train_data), dtype=int)
    test_labels_cnn = np.zeros(len(test_data), dtype=int)

    train_labels_classifier = np.zeros(len(train_data), dtype=int)
    test_labels_classifier = np.zeros(len(test_data), dtype=int)
    converter = np.arange(classes)

    ### classifier#####################################
    for i in range(len(train_labels)):
        train_labels_classifier[i] = np.sum(np.multiply(converter, train_labels[i, :]))
    for j in range(len(test_labels)):
        test_labels_classifier[j] = np.sum(np.multiply(converter, test_labels[j, :]))
    # clf = svm.SVC(kernel='linear', C=1, gamma=0.001, random_state=0, probability=True)
    clf = RandomForestClassifier(n_estimators=1000, random_state=0)
    # srhl_rbf = RBFRandomLayer(n_hidden=50, rbf_width=0.1, random_state=0)
    # clf = GenELMClassifier(hidden_layer=srhl_rbf)
    clf.fit(train_data, train_labels_classifier)
    clf_accuracy = clf.score(test_data, test_labels_classifier)
    print "classifier accuracy = " + str(clf_accuracy)
    #####################################################

    sess.run(tf.global_variables_initializer())
    ### cnn start train##################################
    for epoch in range(epochs):
        if epoch %100 ==0:
            print 'epoch: ' + str(epoch)
        for i in range(len(train_data)):
            train_step.run(feed_dict={xs: train_data.iloc[[i]].values, ys: train_labels[[i]], keep_prob: 0.8})
            loss, acc = sess.run([cross_entropy, accuracy], feed_dict={xs: train_data.iloc[[i]].values,
                                                                          ys: train_labels[[i]],
                                                                          keep_prob: 1.0})
            print(loss, acc)

            train_features_cnn[i] = h_fc1.eval(feed_dict={xs: train_data.iloc[[i]].values})
            train_labels_cnn[i] = np.sum(np.multiply(converter, train_labels[[i]]))


    ### cnn test###
    accuracy = accuracy.eval(feed_dict={xs: test_data, ys: test_labels, keep_prob: 1.0})
    print "conv_net accuracy = " + str(accuracy)

    ### cnn and classifier start train####################
    # for epoch in range(epochs):
    #     for batch in range(len(train_data) // batch_size):
    #         offset = (batch * batch_size) % len(train_data)
    #         train_batch_data = train_data[offset:(offset + batch_size)]
    #         train_batch_labels = train_labels[offset:(offset + batch_size)]
    #
    #         features_batch = h_fc1.eval(feed_dict={xs: train_batch_data})
    #
    #         for j in range(batch_size):
    #             for k in range(L3):
    #                 train_features_cnn[batch_size * batch + j, k] = features_batch[j, k]
    #             train_labels_cnn[batch_size * batch + j] = np.sum(np.multiply(converter, train_batch_labels[j, :]))

    test_features_cnn = h_fc1.eval(feed_dict={xs: test_data})
    for j in range(len(test_data)):
        test_labels_cnn[j] = np.sum(np.multiply(converter, test_labels[j, :]))

    # clf = svm.SVC(kernel='linear', C=1, gamma=0.001, random_state=0, probability=True)
    # clf = RandomForestClassifier(
    #     n_estimators=1000, random_state=0).fit(train_features_cnn, train_labels_cnn)
    # srhl_rbf = RBFRandomLayer(n_hidden=50, rbf_width=0.1, random_state=0)
    # clf = GenELMClassifier(hidden_layer=srhl_rbf)
    clf.fit(train_features_cnn, train_labels_cnn)
    conv_clf_accuracy = clf.score(test_features_cnn, test_labels_cnn)
    print "conv_net classifier accuracy = " + str(conv_clf_accuracy)

    sess.close()


if __name__ == '__main__':
    conv_net_classifier()
