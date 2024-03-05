import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris.csv")

# Split the dataset into features (X) and target (y)
X = df.iloc[:, 1:5]
y = df['Name']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

# Initialisieren und Trainieren des Entscheidungsbaums
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Vorhersagen auf dem Testset
y_pred = clf.predict(X_test)

# Berechnen der Genauigkeit
accuracy = accuracy_score(y_test, y_pred)

# Ausgabe der Genauigkeit
print(f"Genauigkeit des Entscheidungsbaums: {100 * accuracy:.2f}%")

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
tree.plot_tree(decision_tree)

plt.show()
