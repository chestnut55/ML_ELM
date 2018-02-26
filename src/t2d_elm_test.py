import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

import data_util as data
from elm import GenELMClassifier
from random_layer import RBFRandomLayer

A, P, Y, Q = data.t2d_data()

srhl_rbf = RBFRandomLayer(n_hidden=50, rbf_width=0.1, random_state=0)
clf6 = GenELMClassifier(hidden_layer=srhl_rbf).fit(A, Y.values.ravel())
print ("Accuracy of Extreme learning machine Classifier: " + str(clf6.score(P, Q)))

# ==============================================
# plt.figure()
cls = 0
# Set figure size and plot layout
figsize = (20, 15)
f, ax = plt.subplots(1, 1, figsize=figsize)

x = [clf6, 'purple', 'ELM']

y_score = x[0].decision_function(P)
fpr, tpr, _ = roc_curve(Q, y_score)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color=x[1], alpha=0.8,
        label='Test data: {} '
              '(auc = {:.2f})'.format(x[2], roc_auc))

ax.set_xlabel('False Positive Rate', fontsize=15)
ax.set_ylabel('True Positive Rate', fontsize=15)
ax.legend(loc="lower right", fontsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
plt.show()
