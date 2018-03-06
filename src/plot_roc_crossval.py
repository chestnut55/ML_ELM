import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from scipy import interp

import cross_data_util as data
from elm import GenELMClassifier, BaseELM, ELMClassifier
from random_layer import RBFRandomLayer, GRBFRandomLayer
import numpy as np

X, Y = data.colorectal_data()
cv = StratifiedKFold(n_splits=10)

clf = RandomForestClassifier(
    n_estimators=200, random_state=0, criterion='entropy', min_samples_split=20)
scores = cross_val_score(clf, X, Y, cv=cv, scoring='accuracy')
print("Accuracy of Random Forest Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

# algorithm, learning_rate_init, alpha, hidden_layer_sizes
# and activation have impact
clf2 = MLPClassifier(solver='adam', alpha=0.01, max_iter=1000,
                     learning_rate='adaptive', hidden_layer_sizes=(400,),
                     random_state=0, learning_rate_init=1e-2,
                     activation='logistic')
scores = cross_val_score(clf2, X, Y, cv=cv, scoring='accuracy')
print("Accuracy of Multi-layer Perceptron Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf3 = GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
                                  max_depth=10, random_state=0, min_samples_split=5)
scores = cross_val_score(clf3, X, Y, cv=cv, scoring='accuracy')
print("Accuracy of Gradient Boosting Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf4 = SVC(kernel='rbf', C=1,
           gamma=0.001, random_state=0, probability=True)
scores = cross_val_score(clf4, X, Y, cv=cv, scoring='accuracy')
print("Accuracy SVM Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf5 = GaussianNB()
scores = cross_val_score(clf5, X, Y, cv=cv, scoring='accuracy')
print("Accuracy Gaussian Naive Bayes Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

srhl_rbf = RBFRandomLayer(n_hidden=20, rbf_width=0.1, random_state=0)
# srhl_rbf = GRBFRandomLayer(n_hidden=50,random_state=0)
clf6 = GenELMClassifier(hidden_layer=srhl_rbf)
scores = cross_val_score(clf6, X, Y, cv=cv, scoring='accuracy')
print("Accuracy Extreme Learning Machine Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

# ==============================================
f, ax = plt.subplots(1, 1)
params = [(clf, 'red', "Random Forest"),
          (clf2, 'blue', "Multi-layer Perceptron"),
          (clf3, 'green', "Gradient Boosting Trees"),
          (clf4, 'black', "SVM"),
          (clf5, 'orange', 'Gaussian Naive Bayes'),
          (clf6, 'purple', 'Extreme Learning Machine')]

for x in params:
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for train, test in cv.split(X, Y):
        if isinstance(x[0], BaseELM):
            y_true = x[0].fit(X.iloc[train], Y.iloc[train]).decision_function(X.iloc[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(Y.iloc[test], y_true)
            v = interp(mean_fpr, fpr, tpr)
            tprs.append(v)
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        else:
            probas_ = x[0].fit(X.iloc[train], Y.iloc[train]).predict_proba(X.iloc[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(Y.iloc[test], probas_[:, 1])
            v = interp(mean_fpr, fpr, tpr)
            tprs.append(v)
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=x[1], label='{}' '(auc = {:.3f})'.format(x[2], mean_auc), lw=2,
            alpha=.8)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()
