##### Predicting Noah Syndergaard's Pitches #####

#Import libraries for data ingestion and wrangling
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

#Import libraries for data pre-processing, machine learning, and model evaluation
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#Define URL that contains game IDS and the base URL for the pitcher
url = "http://www.brooksbaseball.net/tabs.php?player=592789&p_hand=-1&ppos=-1&cn=200&compType=none&gFilt=&time=month&minmax=ci&var=gl&s_type=2&startDate=03/30/2007&endDate=02/27/2017&balls=-1&strikes=-1&b_hand=-1"
player = "http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=592789&game="

#Scrape the data
def webscraper(url):
    print("Initializing Webscraper")
    print("Creating URLs")  
    
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data)
    
    results=[]
    for gid in soup.find_all('a'):
        results.append(gid.get('href'))
    
    df1 = pd.DataFrame({'results': results})
    df1['id'] = df1.results.str.startswith('http://www.brooksbaseball.net/pfxVB/pfx.php?')
    df1 = df1.loc[df1['id'] == True]
    game_ids = df1['results'].tolist()
    
    results1=[]
    results1 = [i.split('&prevGame=', 1)[1] for i in game_ids]
    urls = list(map('http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=592789&game={0}'.format, results1))
    

    print("Scraping URLs - this takes a while")    
    results2 = []
    for i in urls:
        results2.append(requests.get(i))
        
    results3 = []
    for i in results2:
        results3.append(i.text)
        
    scraped_urls = []
    for i in results3:
        scraped_urls.append(BeautifulSoup(i))
    
    print("Extracting Data")  
    results4 = []
    for i in scraped_urls:
        results4.append(i.findAll('tr'))

    results5 = []
    for j in results4:
        results5.append([[td.getText() for td in j[i].findAll('td')] for i in range(len(j))])
    
    df_list = []
    for i in results5:
        df_list.append(pd.DataFrame(i))
        
    global pitchfx_df
    pitchfx_df = pd.concat(df_list)
    
    print("Success! Collected {} rows of Pitch FX data.".format(pitchfx_df.shape[0]))
    return pitchfx_df

#Wrangle the data    
def wrangling(pitchfx_df):
    print("Wrangling the Pitch FX data.")    
    
    global df    
    df = pitchfx_df[[6, 9, 15, 16, 18, 19, 20, 23, 26]]
    
    df.rename(columns={6: 'Batter', 9: 'Pitch_Outcome', 15: 'Pitch_Type', 16: 'Location', 
                       18: 'Batter_Stance', 19: 'Strikes', 20: 'Balls', 23: 'Outcome', 
                       26: 'Inning'}, inplace=True)
    
    df = df[df.Batter.str.contains("None") == False]
    
    df['Batter'] = df['Batter'].astype('category')
    df['Pitch_Outcome'] = df['Pitch_Outcome'].astype('category')
    df['Pitch_Type'] = df['Pitch_Type'].astype('category')
    df['Location'] = df['Location'].astype('category')
    df['Batter_Stance'] = df['Batter_Stance'].astype('category')
    df['Strikes'] = df['Strikes'].astype('int')
    df['Balls'] = df['Balls'].astype('int')
    df['Outcome'] = df['Outcome'].astype('category')
    df['Inning'] = df['Inning'].astype('category')

    pitches = {'FF': 'Fast', 'SI': 'Fast', 'CU': 'Slow', 'CH': 'Slow', 'SL': 'Slow'}
    df['Pitch_Type'] = df['Pitch_Type'].map(pitches)
    df = df.dropna()

    print("Success! The final dataset contains {} rows of data.".format(df.shape[0]))
    return df
    
#Split data into training and test sets
def data_prep(df):
    print("Preparing data for machine learning.")    
    
    global X
    global Y
    global X_train
    global X_test
    global Y_train
    global Y_test    
       
    X = df[['Batter', 'Pitch_Outcome', 'Location', 'Batter_Stance', 'Outcome', 'Inning']]
    X = pd.get_dummies(X)
          
    Y = df[['Pitch_Type']]
    record = {'Slow': 1, 'Fast': 0}
    Y = Y['Pitch_Type'].map(record)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return

#Run univariate feature selection
def feature_selection(X_train, Y_train):
    print("Selecting top features.")
    test = SelectKBest(k = 4)
    fit = test.fit(X_train, Y_train)

    print("Most predictive features.")
    selected_vars = fit.get_support()
    cols = X.columns.values.tolist()
    cols = pd.DataFrame(cols)
    selected_vars = pd.DataFrame(selected_vars)
    selected = pd.concat([cols, selected_vars], axis = 1)
    print(selected)
    return 

#Conduct grid search to optimize hyperparameters
def grid_search():
    print("Optimizing hyperparamters for SVM")
    svm_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svm_model = svm.SVC()
    supvec = GridSearchCV(svm_model, svm_grid, cv=5, scoring = 'f1')
    supvec.fit(X_train, Y_train)
    print("Best parameters set found for SVM:")
    print(supvec.best_estimator_)
    print()
    
    print("Optimizing hyperparamters for Random Forest")
    rf_grid = {"max_depth": [3, None],
              "n_estimators": [10, 50, 100],
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    rf_model = RandomForestClassifier()
    rf = GridSearchCV(rf_model, rf_grid, cv=5, scoring = 'f1')
    rf.fit(X_train, Y_train)
    print("Best parameters set found for Random Forest:")
    print(rf.best_estimator_)
    print()
    
    #Change grid
    print("Optimizing hyperparamters for AdaBoost")
    ada_grid = {"max_depth": [3, None],
              "n_estimators": [10, 50, 100],
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    ada_model = AdaBoostClassifier()
    ada = GridSearchCV(ada_model, ada_grid, cv=5, scoring = 'f1')
    ada.fit(X_train, Y_train)
    print("Best parameters set found for AdaBoost:")
    print(ada.best_estimator_)
    print()
    
    #Change grid 
    print("Optimizing hyperparamters for Extra Trees")
    trees_grid = {"max_depth": [3, None],
              "n_estimators": [10, 50, 100],
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    trees_model = ExtraTreesClassifier()
    trees = GridSearchCV(ada_model, trees_grid, cv=5, scoring = 'f1')
    trees.fit(X_train, Y_train)
    print("Best parameters set found for Extra Trees:")
    print(trees.best_estimator_)
    print()    
    
    #Change grid
    print("Optimizing hyperparamters for Gradient Boost")
    print("Optimizing hyperparamters for Extra Trees")
    gradient_grid = {"max_depth": [3, None],
              "n_estimators": [10, 50, 100],
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    gradient_model = GradientBoostingClassifier()
    gradient = GridSearchCV(gradient_model, trees_grid, cv=5, scoring = 'f1')
    gradient.fit(X_train, Y_train)
    print("Best parameters set found for Gradient Boosting:")
    print(gradient.best_estimator_)
    print()    
           
    print("Optimizing hyperparamters for K Nearest Neighbors Model")   
    knn_grid = {"n_neighbors": [3, 5, 7, 9],
              "weights": ["uniform", "distance"],
              "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],
              "leaf_size": [30, 40, 50]}
    knn_model = KNeighborsClassifier()
    knn = GridSearchCV(knn_model, knn_grid, cv=5, scoring = 'f1')
    knn.fit(X_train, Y_train)
    print("Best parameters set found for KNN:")
    print(knn.best_estimator_)
    print()
    
    print("Optimizing hyperparamters for Logistic Regression Model")
    log_grid = {"penalty": ["l1", "l2"],
              "C": [1, 10, 100],
              "fit_intercept": [False, True],
              "intercept_scaling": [1, 10, 100],
              "solver": ["newton-cg", "lbfgs", "liblinear"]}
    log_model = LogisticRegression()
    log = GridSearchCV(log_model, log_grid, cv=5, scoring = 'f1')
    log.fit(X_train, Y_train)
    print("Best parameters set found for Logisitc Regression:")
    print()
    print(log.best_estimator_)
    print()
    
    return
    
#Run dummy models to baseline performance
def dummy_model():
    random = DummyClassifier(strategy = 'uniform', random_state = 0)
    random_model = random.fit(X_train, Y_train)
    random_predict = SVM_clf.predict(X_test)
    print('F1 Score for random model:")
    print(f1_score(Y_test, random_predict, average=None))
    print()
    print('Accuracy for random model:')
    print(accuracy_score(Y_test, random_predict))
    
    mostfreq = DummyClassifier(strategy = 'most_frequent', random_state = 0)
    mostfreq_model = mostfreq.fit(X_train, Y_train)
    print('F1 Score for most frequent class model:")
    print(f1_score(Y_test, mostfreq_predict, average=None))
    print()
    print('Accuracy for most frequent class model:')
    print(accuracy_score(Y_test, mostfreq_predict))
    return
    
#Evaluate models using cross-validation
def model_evaluation():
    #Need to edit each model based on gridsearch
    #Need to improve print functions

    SVM_clf = svm.SVC()
    SVM_clf.fit(X_train, Y_train)
    svm_scores = cross_val_score(SVM_clf, X_train, Y_train, cv=10, scoring='f1')
    print(np.mean(svm_scores))
    print()
    print(svm_scores)
    
    RF_clf = RandomForestClassifier()
    RF_clf.fit(X_train, Y_train)
    RF_scores = cross_val_score(RF_clf, X_train, Y_train, cv=10, scoring='f1')
    print(np.mean(RF_scores))
    print()
    print(RF_scores) 
    
    ADA_clf = AdaBoostClassifier()
    ADA_clf.fit(X_train, Y_train)
    ADA_scores = cross_val_score(RF_clf, X_train, Y_train, cv=10, scoring='f1')
    print(np.mean(ADA_scores))
    print()
    print(ADA_scores)   
    
    ETrees_clf = ExtraTreesClassifier()
    ETrees_clf.fit(X_train, Y_train)
    ETrees_scores = cross_val_score(ETrees_clf, X_train, Y_train, cv=10, scoring='f1')
    print(np.mean(ETrees_scores))
    print()
    print(ETrees_scores)

    Gradient_clf = GradientBoostingClassifier()
    Gradient_clf.fit(X_train, Y_train)
    Gradient_scores = cross_val_score(Gradient_clf, X_train, Y_train, cv=10, scoring='f1')
    print(np.mean(Gradient_scores))
    print()
    print(Gradient_scores)  

    KNN_clf = KNeighborsClassifier()
    KNN_clf.fit(X_train, Y_train)
    KNN_scores = cross_val_score(KNN_clf, X_train, Y_train, cv=10, scoring='f1')
    print(np.mean(KNN_scores))
    print()
    print(KNN_scores)  

    Log_clf = log_model = LogisticRegression()
    Log_clf.fit(X_train, Y_train)
    Log_scores = cross_val_score(Log_clf, X_train, Y_train, cv=10, scoring='f1')
    print(np.mean(Log_scores))
    print()
    print(Log_scores) 

    GNB_clf =  GaussianNB()
    GNB_clf.fit(X_train, Y_train)
    GNNB_scores = cross_val_score(GNB_clf, X_train, Y_train, cv=10, scoring='f1')
    print(np.mean(GNB_scores))
    print()
    print(GNB_scores)             

#Confirm model on test set
def test_model():
    svm_predict = SVM_clf.predict(X_test)
    print()
    print('F1 Score for top-performing model')
    print(f1_score(Y_test, svm_predict, average=None))
    print('Classification report for top-performing model')
    print(classification_report(Y_test, svm_predict))

#Execution
webscraper(url)
wrangling(pitchfx_df)
data_prep(df)
feature_selection(X_train, Y_train)
grid_search()
dummy_model()
model_evaluation()
test_model()

    

    
    
    
    
