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

clf = MLPClassifier(hidden_layer_sizes=(3,3,3,3), activation='relu',
                    solver='adam', alpha=0.2, batch_size=100,
                    learning_rate_init=0.01, max_iter=500,
                    verbose=True, momentum=0.1,
                    early_stopping=True)

X_train, X_test, y_train, y_test = train_test_split(X, target)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, y_pred)}\n"
)