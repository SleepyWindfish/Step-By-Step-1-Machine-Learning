import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set=pd.read_csv('Data.csv')

X=data_set.iloc[:,:-1]
y=data_set.iloc[:,-1]
print(X.head)

#Removing the Missing Data
from sklearn.impute import SimpleImputer
impute=SimpleImputer(missing_values=np.nan,strategy='mean')
impute.fit(X.iloc[:,1:3])
X.iloc[:,1:3]=impute.transform(X.iloc[:,1:3])
print("_______________Missing Data___________________")
print(X.head)

#Change String values into numbers that the Machine Learning Algorithom can better coralate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
X=np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)


#Splitting the data into trainting and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Scaling the data- Can't compare apple to oranges!!
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X_train[:,3:]=st.fit_transform(X_train[:,3:])
#Make sure to only transform for the X_test to prevent data leakage since fit will plug in all the vairiables for the formula including the average of the entire data.
X_test[:,3:]=st.transform(X_test[:,3:])
print("______________Done Scaling______________")
print(X_train[:,3:])
