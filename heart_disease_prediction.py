#Importing all necessary libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## Data preprocessing part start from here
# Import the dataset to pandas dataframe
data = pd.read_csv('heart.csv')
data.head()
data.tail()
data.shape
