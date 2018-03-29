#!/usr/bin/python
"""
# Fully Differentiable Deep Neural Decision Forest

[![DOI](https://zenodo.org/badge/20267/chrischoy/fully-differentiable-deep-ndf-tf.svg)](https://zenodo.org/badge/latestdoi/20267/chrischoy/fully-differentiable-deep-ndf-tf)

This repository contains a simple modification of the deep-neural decision
forest [Kontschieder et al.] in TensorFlow. The modification allows joint
optimization of the decision nodes and leaf nodes which theoretically should speed up the training
(haven't verified).


## Motivation:

Deep Neural Deicision Forest, ICCV 2015, proposed an interesting way to incorporate a decision forest into a neural network.

The authors proposed incorporating the terminal nodes of a decision forest as static probability distributions and routing probabilities using sigmoid functions. The final loss is defined as the usual cross entropy between ground truth and weighted average of the terminal probabilities (weights being the routing probabilities).

As there are two trainable parameters, the authors used alternating optimization. They first fixed the terminal node probabilities and trained the base network (routing probabilities), then, fixed the network and optimized the terminal nodes. Such alternating optimization is usually slower than joint optimization since variables that are not being optimized slow down the optimization of the other variable.

However, if we parametrize the terminal nodes using a parametric probability distribution, we can jointly train both terminal and decision nodes, and theoretically, can speed up the convergence.

This code is just a proof-of-concept that

1. One can train both decision nodes and leaf nodes $\pi$ jointly using parametric formulation of leaf (terminal) nodes.

2. It is easy to implement such idea in a symbolic math library.


## Formulation

The leaf node probability $p \in \Delta^{n-1}$ can be parametrized using an $n$ dimensional vector $w_{leaf}$ $\exists w_{leaf}$ s.t. $p = softmax(w_{leaf})$. Thus, we can compute the gradient of $L$ w.r.t $w_{leaf}$ as well and can jointly optimize the terminal nodes as well.

## Experiment

I used a simple (3 convolution + 2 fc) network for this experiment. On the MNIST, it reaches 99.1% after 10 epochs.

## Slides

[SDL Reading Group Slides](https://docs.google.com/presentation/d/1Ze7BAiWbMPyF0ax36D-aK00VfaGMGvvgD_XuANQW1gU/edit?usp=sharing)


## Reference

[Kontschieder et al.] Deep Neural Decision Forests, ICCV 2015

## Slides

[SDL Reading Group Slides](https://docs.google.com/presentation/d/1Ze7BAiWbMPyF0ax36D-aK00VfaGMGvvgD_XuANQW1gU/edit?usp=sharing)

## References
[Kontschieder et al.] Deep Neural Decision Forests, ICCV 2015


## License

The MIT License (MIT)

Copyright (c) 2016 Christopher B. Choy (chrischoy@ai.stanford.edu)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
import numpy as np
import data_load as load
import math
DEPTH = 3  # Depth of a tree
N_LEAF = 2 ** (DEPTH + 1)  # Number of leaf node
N_LABEL = 2  # Number of classes
N_TREE = 5  # Number of trees (ensemble)
N_BATCH = 10  # Number of data points per mini-batch


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))


def model(X, w, w2, w3, w4_e, w_d_e, w_l_e, p_keep_conv, p_keep_hidden):
    """
    Create a forest and return the neural decision forest outputs:

        decision_p_e: decision node routing probability for all ensemble
            If we number all nodes in the tree sequentially from top to bottom,
            left to right, decision_p contains
            [d(0), d(1), d(2), ..., d(2^n - 2)] where d(1) is the probability
            of going left at the root node, d(2) is that of the left child of
            the root node.

            decision_p_e is the concatenation of all tree decision_p's

        leaf_p_e: terminal node probability distributions for all ensemble. The
            indexing is the same as that of decision_p_e.
    """
    assert (len(w4_e) == len(w_d_e))
    assert (len(w4_e) == len(w_l_e))

    l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    d_shape = w4_e[0].get_shape().as_list()[0]
    l3 = tf.reshape(l3, [-1, d_shape])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    decision_p_e = []
    leaf_p_e = []
    for w4, w_d, w_l in zip(w4_e, w_d_e, w_l_e):
        l4 = tf.nn.relu(tf.matmul(l3, w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        decision_p = tf.nn.sigmoid(tf.matmul(l4, w_d))
        leaf_p = tf.nn.softmax(w_l)

        decision_p_e.append(decision_p)
        leaf_p_e.append(leaf_p)

    return decision_p_e, leaf_p_e


##################################################
# Load dataset
##################################################
trX,teX, trY,teY = load.hmp_hmpii_data()
num_input = trX.shape[1]
trX = trX.values.reshape(-1, 1, num_input, 1)
teX = teX.values.reshape(-1, 1, num_input, 1)

# Input X, output Y
X = tf.placeholder("float", [N_BATCH, 1, num_input, 1])
Y = tf.placeholder("float", [N_BATCH, N_LABEL])

##################################################
# Initialize network weights
##################################################
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])

w4_ensemble = []
w_d_ensemble = []
w_l_ensemble = []
for i in range(N_TREE):
    tmp_shape = (int)(math.ceil(num_input / 8.0))
    w4_ensemble.append(init_weights([128 * 1 * tmp_shape, 512]))
    w_d_ensemble.append(init_prob_weights([512, N_LEAF], -1, 1))
    w_l_ensemble.append(init_prob_weights([N_LEAF, N_LABEL], -2, 2))

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

##################################################
# Define a fully differentiable deep-ndf
##################################################
# With the probability decision_p, route a sample to the right branch
decision_p_e, leaf_p_e = model(X, w, w2, w3, w4_ensemble, w_d_ensemble,
                               w_l_ensemble, p_keep_conv, p_keep_hidden)

flat_decision_p_e = []

# iterate over each tree
for decision_p in decision_p_e:
    # Compute the complement of d, which is 1 - d
    # where d is the sigmoid of fully connected output
    decision_p_comp = tf.subtract(tf.ones_like(decision_p), decision_p)

    # Concatenate both d, 1-d
    decision_p_pack = tf.stack([decision_p, decision_p_comp])

    # Flatten/vectorize the decision probabilities for efficient indexing
    flat_decision_p = tf.reshape(decision_p_pack, [-1])
    flat_decision_p_e.append(flat_decision_p)

# 0 index of each data instance in a mini-batch
batch_0_indices = \
    tf.tile(tf.expand_dims(tf.range(0, N_BATCH * N_LEAF, N_LEAF), 1),
            [1, N_LEAF])

###############################################################################
# The routing probability computation
#
# We will create a routing probability matrix \mu. First, we will initialize
# \mu using the root node d, 1-d. To efficiently implement this routing, we
# will create a giant vector (matrix) that contains all d and 1-d from all
# decision nodes. The matrix version of that is decision_p_pack and vectorized
# version is flat_decision_p.
#
# The suffix `_e` indicates an ensemble. i.e. concatenation of all responsens
# from trees.
#
# For depth = 2 tree, the routing probability for each leaf node can be easily
# compute by multiplying the following vectors elementwise.
# \mu =       [d_0,   d_0,   d_0,   d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
# \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
# \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]
#
# Tree indexing
#      0
#    1   2
#   3 4 5 6
##############################################################################
in_repeat = N_LEAF / 2
out_repeat = N_BATCH

# Let N_BATCH * N_LEAF be N_D. flat_decision_p[N_D] will return 1-d of the
# first root node in the first tree.
batch_complement_indices = \
    np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
             * out_repeat).reshape(N_BATCH, N_LEAF)

# First define the routing probabilities d for root nodes
mu_e = []

# iterate over each tree
for i, flat_decision_p in enumerate(flat_decision_p_e):
    mu = tf.gather(flat_decision_p,
                   tf.add(batch_0_indices, batch_complement_indices))
    mu_e.append(mu)

# from the second layer to the last layer, we make the decision nodes
for d in xrange(1, DEPTH + 1):
    indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
    tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                      [1, 2 ** (DEPTH - d + 1)]), [1, -1])
    batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [N_BATCH, 1]))

    in_repeat = in_repeat / 2
    out_repeat = out_repeat * 2

    # Again define the indices that picks d and 1-d for the node
    batch_complement_indices = \
        np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
                 * out_repeat).reshape(N_BATCH, N_LEAF)

    mu_e_update = []
    for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
        mu = tf.multiply(mu, tf.gather(flat_decision_p,
                                  tf.add(batch_indices, batch_complement_indices)))
        mu_e_update.append(mu)

    mu_e = mu_e_update

##################################################
# Define p(y|x)
##################################################
py_x_e = []
for mu, leaf_p in zip(mu_e, leaf_p_e):
    # average all the leaf p
    py_x_tree = tf.reduce_mean(
        tf.multiply(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]),
               tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])), 1)
    py_x_e.append(py_x_tree)

py_x_e = tf.stack(py_x_e)
py_x = tf.reduce_mean(py_x_e, 0)

##################################################
# Define cost and optimization method
##################################################

# cross entropy loss
cost = tf.reduce_mean(-tf.multiply(tf.log(py_x), Y))

# cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(py_x, Y))
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict = tf.argmax(py_x, 1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    # One epoch
    for start, end in zip(range(0, len(trX), N_BATCH), range(N_BATCH, len(trX), N_BATCH)):
        sess.run(train_step, feed_dict={X: trX[start:end], Y: trY[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})

    # Result on the test set
    results = []
    for start, end in zip(range(0, len(teX), N_BATCH), range(N_BATCH, len(teX), N_BATCH)):
        results.extend(np.argmax(teY[start:end], axis=1) ==
                       sess.run(predict, feed_dict={X: teX[start:end], p_keep_conv: 1.0,
                                                    p_keep_hidden: 1.0}))

    print 'Epoch: %d, Test Accuracy: %f' % (i + 1, np.mean(results))
