# Importing the modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importing the dataset
data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, :-1].values
y = data[:, -1].values

# Splitting into train set and test set
X_train, X_test, y_train, y_set = train_test_split(X, y, test_size=0.2, random_state=0)