import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc, roc_curve
from elm import ELMClassifier, GenELMClassifier
from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
from sklearn.cross_validation import train_test_split

if __name__ == "__main__":
    abundance = 'abundance_obesity.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0, dtype=unicode)
    f = f.T
    f.set_index('sampleID', inplace=True)

    # define = '1:disease:obesity'
    # d = pd.DataFrame([s.split(':') for s in define.split(',')])
    # l = pd.DataFrame([0] * len(f))
    # for i in range(len(d)):
    #     tmp = (f[d.iloc[i, 1]].isin(d.iloc[i, 2:])).tolist()
    #     l[tmp] = d.iloc[i, 0]
    #
    # l = l.ix[:,0]
    l = f['disease']

    encoder = LabelEncoder()
    l = pd.Series(encoder.fit_transform(l),
                  index=l.index, name=l.name)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    f = (f - f.min()) / (f.max() - f.min())

    A, P, Y, Q = train_test_split(
        f, l, test_size=0.15, random_state=42)  # Can change to 0.2

    srhl_rbf = RBFRandomLayer(n_hidden=50, rbf_width=0.1, random_state=0)
    clf6 = GenELMClassifier(hidden_layer=srhl_rbf).fit(A, Y.values.ravel())
    print ("Accuracy of Extreme learning machine Classifier: " + str(clf6.score(P, Q)))

    # ==============================================
    plt.figure()
    cls = 0
    # Set figure size and plot layout
    figsize = (20, 15)
    f, ax = plt.subplots(1, 1, figsize=figsize)

    x = [clf6, 'purple', 'ELM']

    # y_true = Q[Q.argsort().index]
    y_score = x[0].decision_function(P)
    # y_prob = x[0].predict_proba(P.ix[Q.argsort().index, :])
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
