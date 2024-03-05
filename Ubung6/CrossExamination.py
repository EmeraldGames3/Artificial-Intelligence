import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def cv(clf, features, classes, n):
    cv_scores = cross_val_score(clf, features, classes, cv=n)
    return cv_scores.mean()


# Load the dataset
df = pd.read_csv("pima-indians-diabetes.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier()

# Perform n-fold cross-validation and print the mean accuracy
mean_accuracy = cv(clf, X, y, 10)
print(f"Mean Accuracy: {mean_accuracy}")
