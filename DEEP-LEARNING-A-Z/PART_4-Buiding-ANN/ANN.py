import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors


raw_data = pd.read_csv('Churn_Modelling.csv')
data = raw_data.copy()
data = data.drop('RowNumber',axis =1)
data = data.drop('CustomerId',axis =1)
data = data.drop('Surname',axis =1)

label_encoder = LabelEncoder()
data['Geography']= label_encoder.fit_transform(data['Geography'])
data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

knn = neighbors.KNeighborsClassifier(n_neighbors = 11)

knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)

accuracy = accuracy_score(y_predict, y_test)
print(accuracy)
