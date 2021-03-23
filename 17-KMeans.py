# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:15:33 2021

@author: dameixne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("C:/Users/dameixne/Documents/Personal/courses/Refactored_Py_DS_ML_Bootcamp-master/17-K-Means-Clustering/College_Data",index_col=0)

# Display all columns when look at dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Examine data
df.head()
df.info()
df.describe()

# scatter plot Room.Board vs Grad.Rate
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

# scatter plot F.Undergad vs Outstate
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

# Histogram of out of state for private and public
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

# Histogram for grad.rate public/private
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

#Why is there a private school with a grade rate higher than 100%?
df[df['Grad.Rate'] > 100]
df.loc[df['Grad.Rate'] > 100,'Grad.Rate'] = 100

# Construct kMeans model
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))
kmeans.cluster_centers_

# Evalutate clusters using labels to see how well it performed
# Convert apply to 0/1
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
    
df['Cluster'] = df['Private'].apply(converter)

df.head()

# Create confusion matrix to see how well it performed
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# What if the data is normlaized? - same results
#from sklearn import preprocessing
#df_normalized = preprocessing.normalize(df.drop(columns='Private'), axis=0)
#df_normalized = pd.DataFrame(df_normalized, columns=df.columns[1:])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Private',axis=1))
df_normalized= scaler.transform(df.drop('Private',axis=1))
df_normalized = pd.DataFrame(df_normalized, columns=df.columns[1:])

kmeans2 = KMeans(n_clusters=2)
kmeans2.fit(df_normalized)
df_normalized['Cluster'] = df['Cluster'].values
print(confusion_matrix(df_normalized['Cluster'],kmeans.labels_))
print(classification_report(df_normalized['Cluster'],kmeans.labels_))
