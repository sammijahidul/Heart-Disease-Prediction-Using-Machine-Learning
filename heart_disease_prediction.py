#Importing all necessary libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## Data preprocessing part start from here
# Import the dataset to pandas dataframe
data = pd.read_csv('heart.csv')

# Check the data information 
data.head()
data.tail()
data.shape
data.info()
data.isnull().sum()
data.describe()
data['target'].value_counts()

# Splitting the output and feature columns[input]
X = data.drop(columns = 'target', axis = 1)
Y = data['target']
print(X)
print(Y)

# Splitting the dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y,
                                                    random_state=2)
print(X.shape, X_train.shape, X_test.shape)

## Started to build the model from here
