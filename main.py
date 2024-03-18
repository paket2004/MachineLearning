import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("IRIS.csv")
# Data preprocessing
# 1. Check for missing values
# https://www.geeksforgeeks.org/working-with-missing-data-in-pandas/#:~:text=In%20order%20to%20check%20missing,are%20True%20for%20NaN%20values
missing_val = pd.isnull(data['sepal_length'])
# print(data[missing_val])
# We don't have any missing values, so we can proceed.

# Split the data into features and outcome
X = data.iloc[:, :4]
y = data.iloc[:, -1]
print(y.shape)
# Encode y-column
encoder = OneHotEncoder()
y.reshape(-1,1)
y = encoder.fit_transform(y)

# print(X)
# Split the data into training and validation sets
