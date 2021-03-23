# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:11:31 2021

@author: dameixne
"""

import numpy as np
import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
yelp = pd.read_csv('C:/Users/dameixne/Documents/Personal/courses/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/yelp.csv')

# Display all columns when look at dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Examine the data
yelp.head()
yelp.info()
yelp.describe()

# Add a length column for number of characters in a review
yelp['text length'] = yelp['text'].apply(len)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# Create some plots for length of review for each rating
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')

sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')

sns.countplot(x='stars',data=yelp,palette='rainbow')

# Examine mean values per column for each start level
stars = yelp.groupby('stars').mean()
stars

# How are features correlated with each other
stars.corr()
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

# Analysis

# Use only 1 or 5-star reviews to make things easier
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

X = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(X)

# Create test/train data and train the model just using feature count
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
                                                    
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train,y_train)

# Make predictions
predictions = nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# Alternative approach with text processing (bow)
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# Train the test split
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)

predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# This performed worse! why?










