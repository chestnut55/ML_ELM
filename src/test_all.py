import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import data_util as data
from elm import GenELMClassifier, BaseELM, ELMClassifier
from random_layer import RBFRandomLayer
import numpy as np

A, P, Y, Q = data.t2d_data()

clf = RandomForestClassifier(
    n_estimators=7000, random_state=0, criterion='entropy', min_samples_split=20).fit(A, Y.values.ravel())
print ("Accuracy of Random Forest Classifier: " + str(clf.score(P, Q)))

# algorithm, learning_rate_init, alpha, hidden_layer_sizes
# and activation have impact
clf2 = MLPClassifier(solver='adam', alpha=0.01, max_iter=1000,
                     learning_rate='adaptive', hidden_layer_sizes=(400,),
                     random_state=0, learning_rate_init=1e-2,
                     activation='logistic').fit(A, Y.values.ravel())
print ("Accuracy of Multi-layer Perceptron Classifier: " + str(clf2.score(P, Q)))

clf3 = GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
                                  max_depth=10, random_state=0, min_samples_split=5).fit(A, Y.values.ravel())
print ("Accuracy of Gradient Boosting Classifier: " + str(clf3.score(P, Q)))

clf4 = SVC(kernel='rbf', C=1,
           gamma=0.001, random_state=0, probability=True).fit(A, Y.values.ravel())
print ("Accuracy of SVM: " + str(clf4.score(P, Q)))

clf5 = GaussianNB().fit(A, Y.values.ravel())
print ("Accuracy of Gaussian Naive Bayes Classifier: " + str(clf5.score(P, Q)))

srhl_rbf = RBFRandomLayer(n_hidden=100, rbf_width=0.1, random_state=0)
clf6 = GenELMClassifier(hidden_layer=srhl_rbf).fit(A, Y.values.ravel())
print ("Accuracy of Extreme learning machine Classifier: " + str(clf6.score(P, Q)))

# ==============================================
cls = 0
# Set figure size and plot layout
figsize = (30, 15)
f, ax = plt.subplots(1, 1, figsize=figsize)

params = [(clf, 'red', "Random Forest"), (clf2, 'blue', "Multi-layer Perceptron"),
          (clf3, 'green', "Gradient Boosting Trees"),
          (clf4, 'black', "SVM"), (clf5, 'orange', 'Gaussian Naive Bayes'), (clf6, 'purple', 'ELM')]

for x in params:
    if isinstance(x[0], BaseELM):
        y_score = x[0].decision_function(P)
        fpr, tpr, _ = roc_curve(Q, y_score)
        roc_auc = auc(fpr, tpr)
    else:
        #y_true = Q[Q.argsort().index]
        #y_prob = x[0].predict_proba(P.ix[Q.argsort().index, :])
        # fpr, tpr, _ = roc_curve(y_true, y_prob[:, cls], pos_label=cls)
        # roc_auc = roc_auc_score(y_true == cls, y_prob[:, cls])
        y_prob = x[0].predict_proba(P)
        fpr, tpr, _ = roc_curve(Q, y_prob[:, cls], pos_label=cls)
        roc_auc = roc_auc_score(Q == cls, y_prob[:, cls])
    ax.plot(fpr, tpr, color=x[1], alpha=0.8,
            label='Test data: {} '
                  '(auc = {:.2f})'.format(x[2], roc_auc))
ax.set_xlabel('False Positive Rate', fontsize=15)
ax.set_ylabel('True Positive Rate', fontsize=15)
ax.legend(loc="lower right", fontsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
plt.show()
