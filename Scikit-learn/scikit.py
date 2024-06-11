import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


#r for relative
titanic_df = pd.read_csv(r"sklearn-code\titanic\titanic.csv")
#print(titanic_df.columns)

# Step 1: Selecting relevant features

x = titanic_df[["Pclass", "Sex", "Age", "Fare",]]
y = titanic_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
