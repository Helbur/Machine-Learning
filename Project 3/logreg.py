import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
np.random.seed(42)

# Preprocessing of data
brc = datasets.load_breast_cancer()
samples, features = brc.data.shape
X = np.hstack((brc.data, np.ones(samples).reshape(-1,1)))
target = brc.target.reshape(-1,1)

clf = LogisticRegression(penalty='l2', dual=False,
                         tol=0.0001, C=1.0, fit_intercept=True,
                         intercept_scaling=1, class_weight=None,
                         random_state=None, solver='lbfgs',
                         max_iter=100, multi_class='auto',
                         verbose=0, warm_start=False,
                         n_jobs=None, l1_ratio=None)

X_train, X_test, y_train, y_test = train_test_split(X, target)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, y_pred)}\n"
)