
#Import libraries
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import csv

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#Change display settings for viewing data
pd.set_option("display.max_columns", 900)
pd.set_option("display.max_rows", 900)

#Read in CSV files of data
offense = pd.read_csv('Offense.csv')
defense = pd.read_csv('Defense.csv')
time = pd.read_csv('Outcomes.csv')

#Join the data
join1 = pd.concat([offense, defense], axis='UID', join='inner')
df = pd.concat([join1, time], axis='UID', join='inner')
df.head()

#Drop UID column
df = df.drop('UID', 1)

#Clean up the labels for home vs away
def label_location (row):
   if row['Location'] == "@" :
      return 'Away'
   else:
    return 'Home'
df['Location'] = df.apply (lambda row: label_location (row), axis=1)

#Extract the W or L from the Rslt column
df['Rslt'] = df['Rslt'].str[:1]

#Extract just the Day from the Day column
df['Day'] = df['Day'].str[:3]

#Extract the month
df['Num'], df['Month'] = df['Month'].str.split('-', 1).str
df = df.drop('Num', 1)

#Create lag variable for wins
df['NextGame'] = df['Rslt'].shift(-1)

#Change the Opp column to be the current opponent instead of the previous game's opponent
df['GameOpp'] = df['Opp'].shift(-1)

#Label the Opp column as Previous Opponent
df.rename(columns={'Opp': 'Prev_Opp'}, inplace=True)

#Label the Rslt column as Previous Rslt
df.rename(columns={'Rslt': 'Prev_Rslt'}, inplace=True)

#Drop NA values
df = df.dropna()

#View head of latest df
df.head(15)

#Get dummies for categorical columns
cat_variables = df[['Off.Thr', 'Month', 'Location', 'Prev_Opp', 'Prev_Rslt', 'Day', 'Time', 'GameOpp']]
cat_variables = pd.get_dummies(cat_variables)
cat_variables.tail()

#Scale numeric variables
from sklearn import preprocessing
num_variables = df[['Off.PA', 'Off.R', 'Off.H', 'Off.2B', 'Off.3B', 'Off.HR', 'Off.RBI', 'Off.BB', 'Off.SO',
                   'Off.HBP', 'Off.GDP', 'Off.SB', 'Off.CS', 'Off.LOB', 'Def.H', 'Def.R', 'Def.BB', 'Def.SO',
                   'Def.HR', 'Def.Pit', 'Def.IR', 'Def.SB', 'Def.CS', 'Def.AB', 'Def.2B', 'Def.3B', 'Def.SF',
                   'Def.GDP', 'Length']]
num_variables = preprocessing.scale(num_variables)
num_variables = pd.DataFrame(num_variables, columns = ['Off.PA', 'Off.R', 'Off.H', 'Off.2B', 'Off.3B', 'Off.HR', 'Off.RBI', 'Off.BB', 'Off.SO',
                   'Off.HBP', 'Off.GDP', 'Off.SB', 'Off.CS', 'Off.LOB', 'Def.H', 'Def.R', 'Def.BB', 'Def.SO',
                   'Def.HR', 'Def.Pit', 'Def.IR', 'Def.SB', 'Def.CS', 'Def.AB', 'Def.2B', 'Def.3B', 'Def.SF',
                   'Def.GDP', 'Length'])
num_variables.tail()

#Join data
X = pd.concat([num_variables, cat_variables], axis = 1)
X = X.dropna()
X.tail()

#Declare target variables
Y = df[['NextGame']]
#Del 794 - 809; for some reason, the preprocessing deleted 15 observations
Y = Y.drop([808, 807, 806, 805, 804, 803, 802, 801, 800, 799, 798, 797, 796, 795, 794])
Y.tail()

#Split into training and test tests
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#Import libraries for variable selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#Select the 20 most powerful predictor variables
test = SelectKBest(k = 20)
fit = test.fit(X_train, Y_train.values.ravel())
print(fit.scores_)

#Print predictor names
vars = fit.get_support()
cols = X.columns.values.tolist()
cols = pd.DataFrame(cols)
vars = pd.DataFrame(vars)
selected = pd.concat([cols, vars], axis = 1)
print(selected)

#Create a new dataframe with only the most predictive variables
df1 = X[['Off.3B', 'Off.BB', 'Def.SO', 'Def.IR', 'Def.SB', 'Def.3B', 'Off.Thr_L', 'Off.Thr_R', 'Month_Aug', 'Prev_Opp_CHW', 
        'Prev_Opp_NYY', 'Prev_Opp_STL', 'Prev_Opp_TOR', 'GameOpp_CHW', 'GameOpp_CIN', 'GameOpp_MIL', 'GameOpp_MIN', 
        'GameOpp_NYY', 'GameOpp_TBR', 'GameOpp_TOR']]

#Recode our Y to 0 or 1 classification
record = {'W': 1, 'L': 0}
Y = Y['NextGame'].map(record)

#Re-split into training and test tests with only the most predictive variables
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(df1, Y)

#Grid search for SVM
from __future__ import print_function
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.metrics import f1_score

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svm_model = svm.SVC()
supvec = GridSearchCV(svm_model, tuned_parameters, cv=5, scoring = 'f1')
supvec.fit(X_train1, Y_train1.values.ravel())

print("Best parameters set found on development set:")
print()
print(supvec.best_estimator_)
print()

#Grid search for Random Forest
from sklearn.ensemble import RandomForestClassifier

param_grid = {"max_depth": [3, None],
              "n_estimators": [10, 50, 100],
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

rf_model = RandomForestClassifier()
rf = GridSearchCV(rf_model, param_grid, cv=5, scoring = 'f1')
rf.fit(X_train1, Y_train1.values.ravel())

print("Best parameters set found on development set:")
print()
print(rf.best_estimator_)
print()

#Grid search for K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

param_grid = {"n_neighbors": [3, 5, 7, 9],
              "weights": ["uniform", "distance"],
              "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],
              "leaf_size": [30, 40, 50]}

knn_model = KNeighborsClassifier()
knn = GridSearchCV(knn_model, param_grid, cv=5, scoring = 'f1')
knn.fit(X_train1, Y_train1.values.ravel())

print("Best parameters set found on development set:")
print()
print(knn.best_estimator_)
print()

#Grid search for Logistic Regression
from sklearn.linear_model import LogisticRegression

param_grid = {"penalty": ["l1", "l2"],
              "C": [1, 10, 100],
              "fit_intercept": [False, True],
              "intercept_scaling": [1, 10, 100],
              "solver": ["newton-cg", "lbfgs", "liblinear"]}

log_model = LogisticRegression()
log = GridSearchCV(log_model, param_grid, cv=5, scoring = 'f1')
log.fit(X_train1, Y_train1.values.ravel())

print("Best parameters set found on development set:")
print()
print(log.best_estimator_)
print()

#Run models and use CV to evaluate

#Fit SVM Model
from sklearn.cross_validation import cross_val_score

SVM_clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.001,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

SVM_clf.fit(X_train1, Y_train1.values.ravel())

svm_scores = cross_val_score(SVM_clf, X_train1, Y_train1.values.ravel(), cv=5, scoring='f1')
print(np.mean(svm_scores))
print()
print(svm_scores)

#Fit Random Forest Model
RF_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=3, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=3, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

RF_clf.fit(X_train1, Y_train1.values.ravel())

RF_scores = cross_val_score(RF_clf, X_train1, Y_train1.values.ravel(), cv=5, scoring='f1')
print(np.mean(RF_scores))
print()
print(RF_scores)

#Fit KNN Model
KNN_clf = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=7, p=2, weights='distance')

KNN_clf.fit(X_train1, Y_train1.values.ravel())

KNN_scores = cross_val_score(KNN_clf, X_train1, Y_train1.values.ravel(), cv=5, scoring='f1')
print(np.mean(KNN_scores))
print()
print(KNN_scores)

#Fit Logistic Regression Model
Log_clf = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=False,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          penalty='l1', random_state=None, solver='newton-cg', tol=0.0001,
          verbose=0)

Log_clf.fit(X_train1, Y_train1.values.ravel())

Log_scores = cross_val_score(Log_clf, X_train1, Y_train1.values.ravel(), cv=5, scoring='f1')
print(np.mean(Log_scores))
print()
print(Log_scores)

#Make predictions on test set using best performing model (SVM)
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

svm_predict = SVM_clf.predict(X_test1)

print(f1_score(Y_test1, svm_predict, average=None))
print()
print(accuracy_score(Y_test1, svm_predict))

############################################################################################################################