import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

#r for relative
titanic_df = pd.read_csv(r"sklearn-code\titanic\titanic.csv")
#print(titanic_df.columns)

# Step 1: Selecting relevant features

from sklearn.model_selection import train_test_split
x = titanic_df[["Pclass", "Sex", "Age", "Fare",]]
y = titanic_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')