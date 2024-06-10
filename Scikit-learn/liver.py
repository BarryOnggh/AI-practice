import pandas as pd

liver_df=pd.read_csv(r"sklearn-code\liver\liver.csv")
print(liver_df.head())

#Cirrhosis classification

#train data
from sklearn.preprocessing import OrdinalEncoder, normalize
from sklearn.model_selection import train_test_split

