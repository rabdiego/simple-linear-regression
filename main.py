# Importing the modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Splitting into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set
y_pred = regressor.predict(X_test)

# Visualizing the test set results
plt.scatter(X_test, y_test, color='c')
plt.plot(X_test, y_pred, color='m')
plt.title('Salary vs Experience', color='m')
plt.xlabel('Years of Experience', color='m')
plt.ylabel('Salary', color='m')
plt.savefig('plot.png')
