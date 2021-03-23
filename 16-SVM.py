# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 14:17:45 2021

@author: dameixne
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

#loans = pd.read_csv("C:/Users/dameixne/Documents/Personal/courses/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/loan_data.csv")

# Display all columns when look at dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

# Look at an example of an Iris image
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# Get the data
iris = sns.load_dataset('iris')

# Examine the data
sns.pairplot(iris,hue='species',palette='Dark2')

# Plot length vs. width of all setosa
setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)

# Evaluate an SVC model
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

svc_model = SVC()
svc_model.fit(X_train,y_train)

predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# This data was pretty easy to model, but still try grid search
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

# print best values
grid.best_params_
grid.best_estimator_
    
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))

# Bonus: 3D Plot of decision planes
import numpy as np
from sklearn import svm, datasets
from mpl_toolkits.mplot3d import Axes3D

X1 = X.iloc[:, :3]
Y = iris['species']

# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# Solve for w3 (z)
z1 = lambda x,y: (-svc_model.intercept_[0]-svc_model._dual_coef_[0][0]*x-
                 svc_model._dual_coef_[0][1]*y) / svc_model._dual_coef_[0][2]
z2 = lambda x,y: (-svc_model.intercept_[1]-svc_model._dual_coef_[1][0]*x-
                 svc_model._dual_coef_[1][1]*y) / svc_model._dual_coef_[1][2]


tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)


fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(X1[Y=='setosa'].iloc[:,0], X1[Y=='setosa'].iloc[:,1], X1[Y=='setosa'].iloc[:,2],'ob')
ax.plot3D(X1[Y=='versicolor'].iloc[:,0], X1[Y=='versicolor'].iloc[:,1], X1[Y=='versicolor'].iloc[:,2],'sr')
ax.plot3D(X1[Y=='virginica'].iloc[:,0], X1[Y=='virginica'].iloc[:,1], X1[Y=='virginica'].iloc[:,2],'sg')
#ax.plot_surface(x, y, z1(x,y))
#ax.plot_surface(x, y, z2(x,y))
ax.view_init(-60, 60)
plt.show()

