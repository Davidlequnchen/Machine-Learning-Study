#!/usr/bin/python3

# this file is the revision for chapter 1 introduction

# setup
import numpy as np 
import os
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import pandas as pd 
import sklearn.linear_model
import sklearn.neighbors


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick',labelsize=12)

#where to save the folder
PROJECT_ROOT_DIR = "."
CHAPTER_ID =  'introducation'
datapath = os.path.join("/home/chenlequn/Machine Learning Study/handson-ml-master/Revision/datasets", "lifesat", "")

def save_fig (fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig (path, format='png', dpi=300)
    
#ignore usless worning
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# prepare the contry statistical data
def prepare_country_stats (oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli [oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values = "Value")
    gdp_per_capita.rename (columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge (left = oecd_bli, right = gdp_per_capita,
                                   left_index = True, right_index = True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0,1,6,8,33,34,35]  # these are the missing data
    keep_indices = list (set(range(36))- set(remove_indices)) # these are the sample data
    return full_country_stats [ ["GDP per capita", 'Life satisfaction']].iloc[keep_indices]



#-----------------------------------------------------------------------------------------
# LOAD DATA 
oecd_bli = pd.read_csv (datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv (datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                              encoding='latin1', na_values="n/a")

# prepare the data
country_stats =  prepare_country_stats (oecd_bli , gdp_per_capita)
X = np.c_[country_stats ["GDP per capita"]]
y = np.c_[country_stats ["Life satisfaction"]]
# numpy.c_ is a way of stacking array, (in colomn order instead of stack behind of another)

#visualization of the data
country_stats.plot (kind = "scatter", x="GDP per capita", y = 'Life satisfaction')
plt.show()

# select a linear model
model = sklearn.linear_model.LinearRegression()

# train the model
model.fit (X, y)

# make a prediction for Cyprus
X_new = [[22587]]
print (model.predict (X_new))

#alternatively can use K Neighbors regress to predict value in this situation
model2 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
model2.fit(X,y)

print (model2.predict(X_new))

#-------------------------hyperparameter tweeking, to get regularized  linear model on the partial data ----
country_stats.plot (kind = "scatter", x="GDP per capita", y = 'Life satisfaction', figsize=(8,3))
ridge = sklearn.linear_model.Ridge(alpha=10**9.5)
Xsample = np.c_[country_stats["GDP per capita"]]
ysample = np.c_[country_stats["Life satisfaction"]]
ridge.fit(Xsample, ysample)
t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0][0]
plt.plot(X, t0ridge + t1ridge * X, "b", label="Regularized linear model on partial data")
plt.show()
