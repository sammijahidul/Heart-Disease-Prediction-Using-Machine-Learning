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