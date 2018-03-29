from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from GCForest import gcForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

n_folds = StratifiedKFold(n_splits=5)
iris = load_iris()
X = iris.data
y = iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

gcf = gcForest(shape_1X=4, window=2, tolerance=0.0)
gcf.fit(X_tr, y_tr)
y_pred = gcf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print("gcForest", acc)

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print("RandomForest", acc)
