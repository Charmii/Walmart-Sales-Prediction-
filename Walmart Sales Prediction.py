# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:17:19 2020

@author: Charmi Shah
"""

#Importing the Dataset
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#Importing the dataset
train_merged = pd.read_csv('Train.csv')
test_merged =pd.read_csv('Test.csv')

print('Merged train Dataset')
train_merged.info()
print('Merged test Dataset')
test_merged.info()

#Replacing missing values in Train_merged with average values of attributes
print('Null values:')
print(train_merged.isnull().sum())
print(test_merged.isnull().sum())
train_merged['MarkDown1']=train_merged['MarkDown1'].transform(lambda x: x.fillna(x.mean()))
train_merged['MarkDown2']=train_merged['MarkDown2'].transform(lambda x: x.fillna(x.mean()))
train_merged['MarkDown3']=train_merged['MarkDown3'].transform(lambda x: x.fillna(x.mean()))
train_merged['MarkDown4']=train_merged['MarkDown4'].transform(lambda x: x.fillna(x.mean()))
train_merged['MarkDown5']=train_merged['MarkDown5'].transform(lambda x: x.fillna(x.mean()))

#Replacing missing values in test_merged with average values of attributes
test_merged['CPI']=test_merged['CPI'].transform(lambda x: x.fillna(x.mean()))
test_merged['Unemployment']=test_merged['Unemployment'].transform(lambda x: x.fillna(x.mean()))
test_merged['Temperature']=test_merged['Temperature'].transform(lambda x: x.fillna(x.mean()))
test_merged['Fuel_Price']=test_merged['Fuel_Price'].transform(lambda x: x.fillna(x.mean()))
test_merged['Size']=test_merged['Size'].transform(lambda x: x.fillna(x.mean()))
test_merged['MarkDown1']=test_merged['MarkDown1'].transform(lambda x: x.fillna(x.mean()))
test_merged['MarkDown2']=test_merged['MarkDown2'].transform(lambda x: x.fillna(x.mean()))
test_merged['MarkDown3']=test_merged['MarkDown3'].transform(lambda x: x.fillna(x.mean()))
test_merged['MarkDown4']=test_merged['MarkDown4'].transform(lambda x: x.fillna(x.mean()))
test_merged['MarkDown5']=test_merged['MarkDown5'].transform(lambda x: x.fillna(x.mean()))

print('Attributes after filling the missing values with the mean')
print(train_merged.isnull().sum())
print(test_merged.isnull().sum())



#Printing the correlations
print('Printing correlation of training set and test set')
train_corr = train_merged.corr()
test_corr = test_merged.corr()
print(train_corr)
print(test_corr)

#Taking care of categorical data
print(train_merged.Type.value_counts())
train_test_data = [train_merged, test_merged]
type_mapping = {"A": 1, "B": 2, "C": 3}
for dataset in train_test_data:
    dataset['Type'] = dataset['Type'].map(type_mapping)
    
type_mapping = {False: 0, True: 1}
for dataset in train_test_data:
    dataset['IsHoliday'] = dataset['IsHoliday'].map(type_mapping)
    
print(train_merged.Type.value_counts())
print(train_merged.IsHoliday.value_counts())


#Classification and accuracy

X = train_merged.drop(columns = ['Weekly_Sales', 'Date'],axis = 1)

test_y = test_merged.drop(columns = ['Date'],axis = 1)

y = train_merged['Weekly_Sales']


#Decision tree Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)
y_pred = regressor.predict(X)
acc_rf= round(regressor.score(X, y) * 100, 2)
print ("Decision Tree Accuracy: %i %% \n"%acc_rf)
y_predicted = regressor.predict(test_y)


#Printing Summary
Summary = pd.DataFrame({
        "Store" : test_merged.Store.astype(str),
        "Department" : test_merged.Dept.astype(str),
        "Date": test_merged.Date.astype(str),
        "Weekly_Sales": y_predicted    }) 
    



