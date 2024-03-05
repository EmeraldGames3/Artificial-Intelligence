import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv("pima-indians-diabetes.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Initialize the classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Support Vector Machine": SVC(),
    "Neural Network": MLPClassifier(max_iter=1000),
}

# Perform 10-fold cross-validation for each classifier
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X, y, cv=10)
    print(f"{name} Accuracy: {cv_scores.mean()}")
