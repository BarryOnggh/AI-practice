import pandas as pd
import numpy as np
from pandasai import SmartDataframe
import os

#turn csv into dataframe

df=pd.read_csv("food_coded.csv")


os.environ['PANDASAI_API_KEY'] = "$2a$10$8sAqSqDsnn4..TMtpRSseO48edkfBQKxWei/P0uoLGnZdIBVfYCBe"
sdf = SmartDataframe(df)
output = sdf.chat("Return the top 3 lowest gpa")
print(output)