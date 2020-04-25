#!/usr/bin/python3

# Revison Code
# Chapter 2: end to end machine learning project
# writer : Chen Lequn


# setup
import numpy as np 
import os
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import pandas as pd 
import sklearn.linear_model
import sklearn.neighbors
import tarfile
from six.moves import urllib


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick',labelsize=12)

#where to save the folder
PROJECT_ROOT_DIR = ".."
CHAPTER_ID =  'introducation'
datapath = os.path.join("../datasets", "lifesat", "")

def save_fig (fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig (path, format='png', dpi=300)
    
#ignore usless worning
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


#----------------------------------STEP1: get the data--------------------------------------------------
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data() # fetch the newst data from URL, extract file from .tgz


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# read the csv file and store it into pandas dataframe
housing = load_housing_data()

print ("\n  this will allow you display first five rows of the data " )
print (housing.head())
print (" \n see the attributes of this data set " )
print (housing.info())
print ("\nuse value_counts to see the content of the text attributes " )
print (housing["ocean_proximity"].value_counts())
print ("\n use describe function to see the content of the numerical attribute " )
print (housing.describe())

# useing matplot to visualize data
housing.hist(bins=50, figsize=(20,15)) # plot histogram, hist function is in pd dataframe object
plt.show()

#----------------split test set and training set------------------------------------------
# to make the random number generation is the same every time,
np.random.seed(42)
# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) #Randomly permute (rearrange) a sequence, or return a permuted range
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print (len(train_set), "tain + ", len(test_set), "test")

#--------------------------solution 2------------------------
# works for both python2 and python3 
from zlib import crc32

def test_set_check1 (identifier, test_ratio):
    # 0xFFFFFFFF is a hexadecimal integer constant. Its decimal value is simply 4294967295 = 2 **32 -1
    # x & y -- "bitwise and". Each bit of the output is 1 if the corresponding bit of x AND of y is 1, otherwise it's 0.
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio *2 **32

def split_train_test_by_id1(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check1(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#----------------------solution 3----------------------
'''
Use instance identifier to decide whether it should go into the test set
E.g. compute a hash of each instance’s identifier, keep only the last byte of the hash, 
and put the instance in the test set if this value is lower or equal to 51(~20% of 256) '''
# this method only support python3 
import hashlib
def test_set_check(identifier, test_ratio, hash):
       # return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio # support only python3
       return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio # support both python2 and python 3

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set] 

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print (test_set.head)

#----------------------solution 4: using sklearn native funtion------
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print (test_set.head) 


#-------------------Stratified Sampling ---------------------------
housing["median_income"].hist() # show the original median income data
plt.show()
# create the income category 
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

print ("\n the income category is created: ")
print (housing["income_cat"].value_counts())
housing["income_cat"].hist()
plt.show()


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# split.split -- based on previously divided income category to perform stratified sampling
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
print (" \n the percentage each category is contributing to the total stratified test sets " )
print ( strat_test_set["income_cat"].value_counts() / len(strat_test_set) )
print ("\n compared: for the total income category, each component percentage")
print ( housing["income_cat"].value_counts() / len(housing) )

# compare the error between each spliting method 
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()

compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
print ("\n shows the percentage of each category for random and stratified sampling, also the error made respectively")
print (compare_props)

#------- cancle the income category property since we no longer need to use
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True) # drop function make the remaining the same
    
    
#-------------------------STEP2: Discover and Visualize the data------------------
# 1
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
# 2
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
# 3
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.show()
# 4--------- 
import matplotlib.image as mpimg

california_img=mpimg.imread("./images/end_to_end_project/california.png")
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.show()

#---------calculate correlation matrix for each attribute-------
corr_matrix = housing.corr()

print  ("\n print the correlation matrix of each attribute")
print ( corr_matrix["median_house_value"].sort_values(ascending=False) )

# plot the correlation matrix
from pandas.plotting import scatter_matrix
# select the repreosentitive one
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# focusing on median income --> the most promising one
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
plt.show()

# ----- create other attributes, and calculate the correlation matrix again ----------------------------------
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
print ("\n new correlation matrix")
print (corr_matrix["median_house_value"].sort_values(ascending=False))


# -----------------------------------------------------------------------------------

# ----------------STEP 3 : Prepare the data for Machine Learning Algorithms------------------------
# first, test set should never be modified, we only study the traning set here
# second, separate the lable and other attributes 
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

#-------------------------------3.1 dealing with missing value-----------------------------------
print ("\n anly value that has missing values" )
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print (sample_incomplete_rows)

# option 1
print ("\n drop the whole subset of data, which total bedroom has NA")
print ( sample_incomplete_rows.dropna(subset=["total_bedrooms"]) )

 # option 2
print ("\n drop the colomn of total bedroom, just get rid of this attribute from dataset")
print (sample_incomplete_rows.drop("total_bedrooms", axis=1))

# option 3---- substitute median value
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) 
print ("\n substitute median value for the missing value --.fillna")
print (sample_incomplete_rows)

# option 4 -----in scikit learn ------------------------------
# from scikitlearn 2.0 --the sklearn.preprocessing.Imputer class was replaced by the sklearn.impute.SimpleImputer class
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy='median')
# Remove the text attribute because median can only be calculated on numerical attributes:
housing_num = housing.drop('ocean_proximity', axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num) 
print ("\n check the value the imputer calculated using .statistics_ function")
print (imputer.statistics_)

# transform the dataset
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
print ("\ncheck out the imputer fixed dataset by substituting the median value")
print (housing_tr.loc[sample_incomplete_rows.index.values])


# ---------------------------3.2 process the categoricla featrue : ocean_proximity------------------
housing_cat = housing[['ocean_proximity']]
print ("\n ocean proximity")
print (housing_cat.head(10))

# use the OrdinalEncoder to encode the string categorical attributs as integers
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
    
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print ("\n Ordinal encoder results for categorical data")
print (housing_cat_encoded[:10])

# Use OneHotEncoder to convert each categorical value to one-hot vector
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print ("\n by default, the OneHotEncode returns a sparse array")
print (housing_cat_1hot.toarray())

#---------------------------3.3 create a custom transformer to add extra attributes-------------------------------
from sklearn.base import BaseEstimator, TransformerMixin

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#------------option 2: using FunctionTransformer class to add attributes-----------
#  Note that we need to set validate=False because the data contains non-float values 
# (validate will default to False in Scikit-Learn 0.22).
from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)

#----------------------- put them into pandas dataframe---------------------
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)

print ("\nthe new data after adding customized attributes")
print (housing_extra_attribs.head())

#-----------------------Build a pipeline for preprocessing the numerical attributes --------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)), # can use CombinedAttiributesAdder() also
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

print ("\n the numerical attributes been preprocessed")
print (housing_num_tr)

#--------------------Pipline for both numerical and categorical attributes----------------------
#----------------method 1: using ColumnTransformer class (preferable)------------------------
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

#------------method 2: using the DataFrameSelector transformer (to just select a subset of the Pandas DataFrame columns), 
#                      and a FeatureUnion:-----------------------------------------
# Create a class to select numerical or categorical columns 
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])
#----------------------------------------------------------------------------
# show the final result for the preprocessed data
old_housing_prepared = old_full_pipeline.fit_transform(housing)


###########################
###---STEP4: Slect and Train a Model
##############################
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# try the full preprocessing pipline for a few instance
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print ("\n print the prediction and actual value for linear regression model testing on a few instance")
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels (actual):", list(some_labels))

# to the whole dataset, make the predictions and calculate the error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print ("\nThe RMSE on the whole dataset: " , lin_rmse)
print ("\nThe Mean Absolute Error (MAE) on the whole dataset: " , lin_mae)


# test another mode: DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print ("\nThe RMSE on the whole dataset by DecisionTreeRegressor: " , tree_rmse)

###########################
###---STEP4: Fine-tune your model
##############################
# the cross validation method
from sklearn.model_selection import cross_val_score
# cv=10: ten folds validation score
# for the DecisionTreeRegressor in previous section
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print ("\n for the cross validaition of DecisionTreeRegressor:")
def display_scores(scores):
    print("\nScores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# for the LinearRegressor in previous section----
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# Try another model: RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# specify n_estimators=10 to avoid a warning about the fact that the default value is going to change to 100 in Scikit-Learn 0.22
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print ("\n the RMSE for the prediction made by RandomForestRegressor", forest_rmse)

#------cross validation the model---------
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores) # we are using minus sign here
display_scores(forest_rmse_scores)


# Try Another model: svm.
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print ("\n the RMSE for the prediction made by SVM", svm_rmse)


##############
##GridSearchCV help you to search !
'''
Tell which hyperparameter, and what values to try out
It will evaluate all possible values, using cross-validation
E.g. search the best combination of hyperparameter values
from sklearn.model_selection import GridSearchCV'''
from sklearn.model_selection import GridSearchCV

###Example here: for RandomForestRegressor, find the best hyperparameter value
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
print ("\n print out the best hyperparameter found:", grid_search.best_params_)

#look at the score of each hyperparameter combination tested during the grid search:
cvres = grid_search.cv_results_
print ("\nthe score of each hyperparameter combination tested during the grid search")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
    
#####When hyperparameter search space is large use RandomizedSearchCV instead
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
# it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
# number of iterations: 10
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_
print ("\nthe score of each hyperparameter combination tested during the RandomizedSearchCV")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
#-------------------------Analyze the best models and their errors-----
# E.g. RandomForestRegressor can indicate the relative importance of each attribute for making accurate predictions
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


#-------------final model-----------------
final_model = grid_search.best_estimator_
# validation on test sets
# separaet the label and attributes
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print ("\n the final model RMSE is:", final_rmse)

# compute a 95% confidence interval for the test RMSE:
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))

# or, do it mannually
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)
# Alternatively, we could use a z-scores rather than t-scores:
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)



################## A full pipeline with both preparation and prediction
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)

##############
#####Model persistence using joblib
my_model = full_pipeline_with_predictor
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF

my_model_loaded = joblib.load("my_model.pkl") # DIFF