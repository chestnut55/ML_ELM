import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as prep


class class_metrics:
    def __init__(self):
        self.accuracy = []
        self.f1 = []
        self.precision = []
        self.recall = []
        self.auc = []
        self.roc_curve = []
        self.confusion_matrix = []


class feature_importance:
    def __init__(self, feat, p):
        self.feat_sel = feat
        self.imp = np.array([p] * len(feat))


def compute_feature_importance(el, feat, feat_sel, ltype):
    fi = feature_importance(feat, 0.0)
    if ltype == 'rf':
        t = el.feature_importances_
    elif (ltype == 'lasso') | (ltype == 'enet'):
        t = abs(el.coef_) / sum(abs(el.coef_))
    else:
        t = [1.0 / len(feat_sel)] * len(feat_sel)
    ti = [feat.index(s) for s in feat_sel]
    fi.imp[ti] = t

    t = sorted(range(len(t)), key=lambda s: t[s], reverse=True)
    fi.feat_sel = [feat[ti[s]] for s in t if fi.imp[ti[s]] != 0]

    return fi


def save_results(l, l_es, p_es, i_tr, i_u, nf, runs_n, runs_cv_folds):
    n_clts = len(np.unique(l))
    cm = class_metrics()

    if out_f:
        fidoutes.write('#features\t' + str(nf) + '\n')
        if n_clts == 2:
            fidoutroc.write('#features\t' + str(nf) + '\n')

    for j in range(runs_n * runs_cv_folds):
        l_ = pd.DataFrame([l.loc[i] for i in l[~i_tr[j] & i_u[j / runs_cv_folds]].index]).values.flatten().astype('int')
        l_es_ = l_es[j].values.flatten().astype('int')
        p_es_pos_ = p_es[j].loc[:, 1].values
        ii_ts_ = [i for i in range(len(i_tr[j])) if i_tr[j][i] == False]

        cm.accuracy.append(metrics.accuracy_score(l_, l_es_))
        cm.f1.append(metrics.f1_score(l_, l_es_, pos_label=None, average='weighted'))
        cm.precision.append(metrics.precision_score(l_, l_es_, pos_label=None, average='weighted'))
        cm.recall.append(metrics.recall_score(l_, l_es_, pos_label=None, average='weighted'))
        if len(np.unique(l_)) == n_clts:
            if n_clts == 2:
                cm.auc.append(metrics.roc_auc_score(l_, p_es_pos_))
                cm.roc_curve.append(metrics.roc_curve(l_, p_es_pos_))
                fidoutroc.write('run/fold\t' + str(j / runs_cv_folds) + '/' + str(j % runs_cv_folds) + '\n')
                for i in range(len(cm.roc_curve[-1])):
                    for i2 in range(len(cm.roc_curve[-1][i])):
                        fidoutroc.write(str(cm.roc_curve[-1][i][i2]) + '\t')
                    fidoutroc.write('\n')
            cm.confusion_matrix.append(metrics.confusion_matrix(l_, l_es_, labels=np.unique(l.astype('int'))))

        if out_f:
            fidoutes.write('run/fold\t' + str(j / runs_cv_folds) + '/' + str(j % runs_cv_folds))
            fidoutes.write('\ntrue labels\t')
            [fidoutes.write(str(i) + '\t') for i in l_]
            fidoutes.write('\nestimated labels\t')
            [fidoutes.write(str(i) + '\t') for i in l_es_]
            if n_clts <= 2:
                fidoutes.write('\nestimated probabilities\t')
                [fidoutes.write(str(i) + '\t') for i in p_es_pos_]
            fidoutes.write('\nsample index\t')
            [fidoutes.write(str(i) + '\t') for i in ii_ts_]
            fidoutes.write('\n')

    fidout.write('#samples\t' + str(sum(sum(i_u)) / len(i_u)))
    fidout.write('\n#features\t' + str(nf))
    fidout.write('\n#runs\t' + str(runs_n))
    fidout.write('\n#runs_cv_folds\t' + str(runs_cv_folds))

    fidout.write('\naccuracy\t' + str(np.mean(cm.accuracy)) + '\t' + str(np.std(cm.accuracy)))
    fidout.write('\nf1\t' + str(np.mean(cm.f1)) + '\t' + str(np.std(cm.f1)))
    fidout.write('\nprecision\t' + str(np.mean(cm.precision)) + '\t' + str(np.std(cm.precision)))
    fidout.write('\nrecall\t' + str(np.mean(cm.recall)) + '\t' + str(np.std(cm.recall)))
    if n_clts == 2:
        fidout.write('\nauc\t' + str(np.mean(cm.auc)) + '\t' + str(np.std(cm.auc)))
    else:
        fidout.write('\nauc\t[]\t[]')
    fidout.write('\nconfusion matrix')
    if len(cm.confusion_matrix) > 0:
        for i in range(len(cm.confusion_matrix[0])):
            for i2 in range(len(cm.confusion_matrix[0][i])):
                fidout.write(
                    '\t' + str(np.sum([cm.confusion_matrix[j][i][i2] for j in range(len(cm.confusion_matrix))])))
            fidout.write('\n')
    else:
        fidout.write('\n')

    return cm


if __name__ == "__main__":
    abundance = 'abundance_obesity.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0, dtype=unicode)
    f = f.T

    # out_f = 'results/abundance_hmp-hmpii__d-bodysite__l-rf__u-subjectID'
    out_f = 'results/abundance_obesity__d-disease__l-rf'
    fidout = open(out_f + '.txt', 'w')
    fidoutes = open(out_f + '_estimations.txt', 'w')
    fidoutroc = open(out_f + '_roccurve.txt', 'w')

    # unique = 'subjectID'
    unique = None
    # pf = pd.DataFrame([s.split(':') for s in unique.split(',')])

    # define = '1:bodysite:stool,2:bodysite:anterior_nares,3:bodysite:l_retroauricular_crease:r_retroauricular_crease,4:bodysite:mid_vagina:posterior_fornix:vaginal_introitus'
    define = '1:disease:obesity'
    d = pd.DataFrame([s.split(':') for s in define.split(',')])
    l = pd.DataFrame([0] * len(f))
    for i in range(len(d)):
        l1 = d.iloc[i, 1]
        l2 = d.iloc[i, 2:]
        tmp = f[l1].isin(l2)
        tmp1 = tmp.tolist()
        l[tmp1] = d.iloc[i, 0]

    runs_cv_folds = 10
    runs_n = 20
    i_tr = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n * runs_cv_folds))
    # i_u = pd.DataFrame(False, index=range(len(f.index)), columns=range(runs_n))
    # meta_u = [s for s in f.columns if s in pf.iloc[0, 0:].tolist()]
    i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n))

    set_seed = True
    for j in range(runs_n):
        if set_seed:
            np.random.seed(j)

        # ii_u = [s - 1 for s in (f.loc[np.random.permutation(f.index), :].drop_duplicates(meta_u)).index]
        # i_u[j][ii_u] = True
        ii_u = range(len(f.index))

        skf = StratifiedKFold(l.iloc[i_u.values.T[j], 0], runs_cv_folds, shuffle=True, random_state=j)
        for i in range(runs_cv_folds):
            for s in np.where(skf.test_folds == i)[0]:
                i_tr[j * runs_cv_folds + i][ii_u[s]] = False
    i_tr = i_tr.values.T
    i_u = i_u.values.T

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    if 'unclassified' in f.columns: feat.append('unclassified')
    f = f.loc[:, feat].astype('float')
    f = (f - f.min()) / (f.max() - f.min())

    # random forest
    fi = []
    clf = []
    p_es = []
    l_es = []

    for j in range(runs_n * runs_cv_folds):
        fi.append(feature_importance(feat, 1.0 / len(feat)))
        # random forest
        rf_clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1)
        x_train = f.loc[i_tr[j] & i_u[j / runs_cv_folds], fi[j].feat_sel].values
        y_train = l[i_tr[j] & i_u[j / runs_cv_folds]].values.flatten().astype('int')
        rf_clf.fit(x_train, y_train)
        clf.append(rf_clf)

        p_es.append(
            pd.DataFrame(clf[j].predict_proba(f.loc[~i_tr[j] & i_u[j / runs_cv_folds], fi[j].feat_sel].values)))
        l_es.append(
            pd.DataFrame([list(p_es[j].iloc[i, :]).index(max(p_es[j].iloc[i, :])) for i in range(len(p_es[j]))]))

    cm = save_results(l, l_es, p_es, i_tr, i_u, len(feat), runs_n, runs_cv_folds)

    # fi_f = []
    # for j in range(runs_n * runs_cv_folds):
    #     fi_f.append(compute_feature_importance(clf[j], feat, fi[j].feat_sel, 'rf'))
    #
    # cv_grid = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
    # for k in cv_grid:
    #     clf_f = []
    #     p_es_f = []
    #     l_es_f = []
    #     for j in range(runs_n * runs_cv_folds):
    #         fi.append(feature_importance(feat, 1.0 / len(feat)))
    #         clf_f.append(
    #             RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1).fit(
    #                 f.loc[i_tr[j] & i_u[j / runs_cv_folds], fi_f[j].feat_sel[:k]].values,
    #                 l[i_tr[j] & i_u[j / runs_cv_folds]].values.flatten().astype('int')))
    #         p_es_f.append(pd.DataFrame(
    #             clf_f[j].predict_proba(f.loc[~i_tr[j] & i_u[j / runs_cv_folds], fi_f[j].feat_sel[:k]].values)))
    #         l_es_f.append(pd.DataFrame(
    #             [list(p_es_f[j].iloc[i, :]).index(max(p_es_f[j].iloc[i, :])) for i in range(len(p_es_f[j]))]))
    #     cm_f = save_results(l, l_es_f, p_es_f, i_tr, i_u, k, runs_n, runs_cv_folds)

    fidout.close()
    fidoutes.close()
    fidoutroc.close()
