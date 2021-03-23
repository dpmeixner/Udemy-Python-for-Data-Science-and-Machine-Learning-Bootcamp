# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 07:52:36 2021

@author: dameixne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("C:/Users/dameixne/Documents/Personal/courses/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression\Ecommerce Customers")

# Display all columns when look at dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
    
# Examine customers
customers.head()
customers.info()
customers.describe()

#did this earlier from CLI, didn't finish writing in file