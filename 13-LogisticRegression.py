# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:10:36 2021

@author: dameixne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("C:/Users/dameixne/Documents/Personal/courses/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv")

# Display all columns when look at dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

# explorte the data
train.head()

# Explore missing data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Plot total number of survivors, by sex, class
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

# Plot distribution of passengers by age, sibling/spouse count, fare
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

train['Age'].hist(bins=30,color='darkred',alpha=0.7)

sns.countplot(x='SibSp',data=train)

train['Fare'].hist(color='green',bins=40,figsize=(8,4))

# Use average age by class to fill in missing age data
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

# This is the implementation from the lab
'''
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
'''

# Alternative cleaner implementation is
train["Age"] = train.groupby("Pclass")['Age'].transform(
    lambda x: x.fillna(x.mean()))

# Confirm age data doesn't have any NAs
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Remove cabin column - not enough data to be relevant
train.drop('Cabin',axis=1,inplace=True)
train.head()

# Remove remaining na data (which is just two rows with an NA for Embarked)
#train[train.isna().any(axis=1)]
train.dropna(inplace=True)

# Convert non-numeric columns to categorical data
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()


# Build a model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))
