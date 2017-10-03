# Library imports
from helpers import *

import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, Imputer

# Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Logistic Regression, SVM, MLP
def train_classification(X_train, X_test, Y_train, Y_test, X_train_full):
    # Random Forest
    print('training random forest')
    rf_pipe = Pipeline([('scaler', MinMaxScaler()),
                        ('classifier', RandomForestClassifier())])

    tree_param_grid = {'classifier__n_estimators': [100],
                       'classifier__max_features': ['auto', 'sqrt', 'log2'],
                       'classifier__min_samples_split': [2, 3, 4, 5, 8, 10],
                       'classifier__class_weight': [None, 'balanced_subsample',
                                                    {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4},
                                                    {0: 1, 1: 5}, {0: 1, 1: 6}, {0: 1, 1: 7}]}

    rf_search = GridSearchCV(rf_pipe, param_grid=tree_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
    rf_search.fit(X_train, Y_train)

    rf_eval_pipe = make_pipeline(MinMaxScaler(), rf_search.best_estimator_)
    rf_search_best_cv = cross_val_score(rf_eval_pipe,
                                        X_train, Y_train,
                                        cv=10,
                                        scoring='roc_auc',
                                        n_jobs=-1)

    rf_data = [
        {'model': rf_search.best_estimator_.named_steps['classifier'],
         'mean_cv_scores': rf_search_best_cv.mean()}]

    rf_model_results_df = pd.DataFrame(rf_data)
    best_rf = rf_model_results_df.loc[rf_model_results_df['mean_cv_scores'].idxmax()]
    best_rf.to_csv('best_random_forest.csv', index=False)

    rf_clf = best_rf.model
    rf_clf.fit(X_train_full, Y_train)
    filename = 'best_rf_model.sav'
    pickle.dump(rf_clf, open(filename, 'wb'))

    rf_predict = rf_clf.predict_proba(X_test)
    rf_roc_auc = roc_auc_score(Y_test, rf_predict[:, 1])

    rf_test_set = [
        {'roc_auc_score_on_test_set': rf_roc_auc}]

    rf_test_set = pd.DataFrame(rf_test_set)
    rf_test_set.to_csv('rf_results_on_test_set.csv', index=False)

    # Extra Trees
    print('training extra trees')
    et_pipe = Pipeline([('scaler', MinMaxScaler()),
                        ('classifier', ExtraTreesClassifier())])

    et_search = GridSearchCV(et_pipe, param_grid=tree_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
    et_search.fit(X_train, Y_train)

    et_eval_pipe = make_pipeline(MinMaxScaler(), et_search.best_estimator_)
    et_search_best_cv = cross_val_score(et_eval_pipe,
                                        X_train, Y_train,
                                        cv=10,
                                        scoring='roc_auc',
                                        n_jobs=-1)

    et_data = [
        {'model': et_search.best_estimator_.named_steps['classifier'],
         'mean_cv_scores': et_search_best_cv.mean()}]

    et_model_results_df = pd.DataFrame(et_data)
    best_et = et_model_results_df.loc[et_model_results_df['mean_cv_scores'].idxmax()]
    best_et.to_csv('best_extra_trees.csv', index=False)

    et_clf = best_et.model
    et_clf.fit(X_train_full, Y_train)
    filename = 'best_et_model.sav'
    pickle.dump(et_clf, open(filename, 'wb'))

    et_predict = et_clf.predict_proba(X_test)
    et_roc_auc = roc_auc_score(Y_test, et_predict[:, 1])

    et_test_set = [
        {'roc_auc_score_on_test_set': et_roc_auc}]

    et_test_set = pd.DataFrame(et_test_set)
    et_test_set.to_csv('et_results_on_test_set.csv', index=False)

    # Gradient Boosting
    print('training gradient boosting')
    gb_pipe = Pipeline([('scaler', MinMaxScaler()),
                        ('classifier', GradientBoostingClassifier())])

    gb_param_grid = {'classifier__learning_rate': [0.01, 0.1, 1, 10],
                     'classifier__n_estimators': [10, 25, 50, 75, 100],
                     'classifier__max_depth': [2, 3, 4, 5, 6]}

    gb_search = GridSearchCV(gb_pipe, param_grid=gb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    gb_search.fit(X_train, Y_train)

    gb_eval_pipe = make_pipeline(MinMaxScaler(), gb_search.best_estimator_)
    gb_search_best_cv = cross_val_score(gb_eval_pipe,
                                        X_train, Y_train,
                                        cv=10,
                                        scoring='roc_auc',
                                        n_jobs=-1)

    gb_data = [
        {'model': gb_search.best_estimator_.named_steps['classifier'],
         'mean_cv_scores': gb_search_best_cv.mean()}]

    gb_model_results_df = pd.DataFrame(gb_data)
    best_gb = gb_model_results_df.loc[gb_model_results_df['mean_cv_scores'].idxmax()]
    best_gb.to_csv('best_gradient_boosting.csv', index=False)

    gb_clf = best_gb.model
    gb_clf.fit(X_train_full, Y_train)
    filename = 'best_gb_model.sav'
    pickle.dump(gb_clf, open(filename, 'wb'))

    gb_predict = gb_clf.predict_proba(X_test)
    gb_roc_auc = roc_auc_score(Y_test, gb_predict[:, 1])

    gb_test_sgb = [
        {'roc_auc_score_on_test_set': gb_roc_auc}]

    gb_test_sgb = pd.DataFrame(gb_test_sgb)
    gb_test_sgb.to_csv('gb_results_on_test_set.csv', index=False)

    # AdaBoost
    print('training ada boost')
    ada_pipe = Pipeline([('scaler', MinMaxScaler()),
                         ('classifier', AdaBoostClassifier())])

    ada_param_grid = {"classifier__learning_rate": [0.001, 0.01, 0.01, 0.1, 1, 10, 10],
                      "classifier__n_estimators": [10, 25, 50, 75, 100, 200]}

    ada_search = GridSearchCV(ada_pipe, param_grid=ada_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
    ada_search.fit(X_train, Y_train)

    ada_eval_pipe = make_pipeline(MinMaxScaler(), ada_search.best_estimator_)
    ada_search_best_cv = cross_val_score(ada_eval_pipe,
                                         X_train, Y_train,
                                         cv=10,
                                         scoring='roc_auc',
                                         n_jobs=-1)

    ada_data = [
        {'model': ada_search.best_estimator_.named_steps['classifier'],
         'mean_cv_scores': ada_search_best_cv.mean()}]

    ada_model_results_df = pd.DataFrame(ada_data)
    best_ada = ada_model_results_df.loc[ada_model_results_df['mean_cv_scores'].idxmax()]
    best_ada.to_csv('best_ada_boost.csv', index=False)

    ada_clf = best_ada.model
    ada_clf.fit(X_train_full, Y_train)
    filename = 'best_ada_model.sav'
    pickle.dump(ada_clf, open(filename, 'wb'))

    ada_predict = ada_clf.predict_proba(X_test)
    ada_roc_auc = roc_auc_score(Y_test, ada_predict[:, 1])

    ada_test_set = [
        {'roc_auc_score_on_test_set': ada_roc_auc}]

    ada_test_set = pd.DataFrame(ada_test_set)
    ada_test_set.to_csv('ada_results_on_test_set.csv', index=False)

    # Logistic Regression
    logreg_pipe = Pipeline([('scaler', MinMaxScaler()),
                            ('classifier', LogisticRegression())])

    logreg_param_grid = {'classifier__C': [0.01, 0.1, 1, 10],
                         'classifier__class_weight': [None, 'balanced',
                                                      {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4},
                                                      {0: 1, 1: 5}, {0: 1, 1: 6}, {0: 1, 1: 7}],
                         'classifier__penalty': ['l1', 'l2'],
                         'classifier__fit_intercept': [True, False]}

    logreg_search = GridSearchCV(logreg_pipe, param_grid=logreg_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
    logreg_search.fit(X_train, Y_train)

    logreg_eval_pipe = make_pipeline(MinMaxScaler(), logreg_search.best_estimator_)
    logreg_search_best_cv = cross_val_score(logreg_eval_pipe,
                                            X_train, Y_train,
                                            cv=10,
                                            scoring='roc_auc',
                                            n_jobs=-1)

    logreg_data = [
        {'model': logreg_search.best_estimator_.named_steps['classifier'],
         'mean_cv_scores': logreg_search_best_cv.mean()}]

    logreg_model_results_df = pd.DataFrame(logreg_data)
    best_logreg = logreg_model_results_df.loc[logreg_model_results_df['mean_cv_scores'].idxmax()]
    best_logreg.to_csv('best_log_reg.csv', index=False)

    logreg_clf = best_logreg.model
    logreg_clf.fit(X_train_full, Y_train)
    filename = 'best_logreg_model.sav'
    pickle.dump(logreg_clf, open(filename, 'wb'))

    logreg_predict = logreg_clf.predict_proba(X_test)
    logreg_roc_auc = roc_auc_score(Y_test, logreg_predict[:, 1])

    logreg_test_logreg = [
        {'roc_auc_score_on_test_logreg': logreg_roc_auc}]

    logreg_test_logreg = pd.DataFrame(logreg_test_logreg)
    logreg_test_logreg.to_csv('logreg_results_on_test_set.csv', index=False)

    # SVM
    svm_pipe = Pipeline([('scaler', MinMaxScaler()),
                         ('classifier', SVC())])

    svm_param_grid = [{'classifier__kernel': ['rbf', 'poly'],
                       'classifier__gamma': [0.01, 0.1, 1, 10],
                       'classifier__C': [0.01, 0.1, 1, 10],
                       'classifier__probability': [True]},

                    {'classifier__kernel': ['linear'],
                     'classifier__C': [0.01, 0.1, 1, 10],
                     'classifier__probability': [True]}]

    svm_search = GridSearchCV(svm_pipe, param_grid=svm_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
    svm_search.fit(X_train, Y_train)

    svm_eval_pipe = make_pipeline(MinMaxScaler(), svm_search.best_estimator_)
    svm_search_best_cv = cross_val_score(svm_eval_pipe,
                                         X_train, Y_train,
                                         cv=10,
                                         scoring='roc_auc',
                                         n_jobs=-1)

    svm_data = [
        {'model': svm_search.best_estimator_.named_steps['classifier'],
         'mean_cv_scores': svm_search_best_cv.mean()}]

    svm_model_results_df = pd.DataFrame(svm_data)
    best_svm = svm_model_results_df.loc[svm_model_results_df['mean_cv_scores'].idxmax()]
    best_svm.to_csv('best_svm.csv', index=False)

    svm_clf = best_svm.model
    svm_clf.fit(X_train_full, Y_train)
    filename = 'best_svm_model.sav'
    pickle.dump(svm_clf, open(filename, 'wb'))

    svm_predict = svm_clf.predict_proba(X_test)
    svm_roc_auc = roc_auc_score(Y_test, svm_predict[:, 1])

    svm_test_set = [
        {'roc_auc_score_on_test_set': svm_roc_auc}]

    svm_test_svm = pd.DataFrame(svm_test_set)
    svm_test_svm.to_csv('svm_results_on_test_set.csv', index=False)

    # Perceptron
    mlp_pipe = Pipeline([('scaler', MinMaxScaler()),
                         ('classifier', MLPClassifier())])

    mlp_param_grid = {'classifier__activation': ['identity', 'tanh', 'relu'],
                       'classifier__solver': ['lbfgs', 'sgd', 'adam'],
                       'classifier__alpha': [0.0001, 0.01, 1, 10],
                       'classifier__learning_rate': ['constant', 'invscaling', 'adaptive']}

    mlp_search = GridSearchCV(mlp_pipe, param_grid=mlp_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
    mlp_search.fit(X_train, Y_train)

    mlp_eval_pipe = make_pipeline(MinMaxScaler(), mlp_search.best_estimator_)
    mlp_search_best_cv = cross_val_score(mlp_eval_pipe,
                                         X_train, Y_train,
                                         cv=10,
                                         scoring='roc_auc',
                                         n_jobs=-1)

    mlp_data = [
        {'model': mlp_search.best_estimator_.named_steps['classifier'],
         'mean_cv_scores': mlp_search_best_cv.mean()}]

    mlp_model_results_df = pd.DataFrame(mlp_data)
    best_mlp = mlp_model_results_df.loc[mlp_model_results_df['mean_cv_scores'].idxmax()]
    best_mlp.to_csv('best_mlp.csv', index=False)

    mlp_clf = best_mlp.model
    mlp_clf.fit(X_train_full, Y_train)
    filename = 'best_mlp_model.sav'
    pickle.dump(mlp_clf, open(filename, 'wb'))

    mlp_predict = mlp_clf.predict_proba(X_test)
    mlp_roc_auc = roc_auc_score(Y_test, mlp_predict[:, 1])

    mlp_test_set = [
        {'roc_auc_score_on_test_set': mlp_roc_auc}]

    mlp_test_mlp = pd.DataFrame(mlp_test_set)
    mlp_test_mlp.to_csv('mlp_results_on_test_set.csv', index=False)
    return None

if __name__ == "__main__":
    # Data In
    hof, batting, master, pitching, fielding, awards, all_stars, postseason, \
    world_series_and_cs = hall_of_fame_ingestion_and_wrangling()

    mitchell_players = run_webscraper_for_mitchell_report()
    suspended_players = run_webscraper_for_suspended_players()

    # Helpers
    df, df2 = clean_data(hof, batting, master, pitching, fielding, postseason, awards, all_stars, world_series_and_cs)
    df, df2 = add_players_connected_with_steroids(df, df2, mitchell_players, suspended_players)
    df.to_csv('full_hof_data.csv')
    df2.to_csv('full_hof_data_all_players.csv')

    X_train, X_test, Y_train, Y_test = machine_learning_prep_classification(df)
    X_train_full = create_scaled_copy_of_X_train(X_train)
    X_train.to_csv('X_train.csv')

    # Classification
    train_classification(X_train, X_test, Y_train, Y_test, X_train_full)
