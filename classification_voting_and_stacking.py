# Library imports
from helpers import *

import pandas as pd
import pickle

from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# Voting classifier based on strong models
def run_voting_classifier(X_train, X_test, Y_train, Y_test, X_train_full):
    # Load models
    ada_clf = pickle.load(open('best_ada_model.sav', 'rb'))
    et_clf = pickle.load(open('best_et_model.sav', 'rb'))
    gb_clf = pickle.load(open('best_gb_model.sav', 'rb'))
    lr_clf = pickle.load(open('best_logreg_model.sav', 'rb'))
    mlp_clf = pickle.load(open('best_mlp_model.sav', 'rb'))
    rf_clf = pickle.load(open('best_rf_model.sav', 'rb'))
    svm_clf = pickle.load(open('best_svm_model.sav', 'rb'))

    # Voting Classifier 1 - gradient boosting, logistic regression, random forest
    print('running voting classifier #1')
    voting_classifier1 = VotingClassifier(estimators=[
        ('ada_boost', ada_clf), ('extra_trees', et_clf), ('gradient_boosting', gb_clf), ('log regg', lr_clf),
            ('perceptron', mlp_clf), ('random_forest', rf_clf), ('svm', svm_clf)], voting='soft')

    voting_classifier1_pipeline = make_pipeline(MinMaxScaler(), VotingClassifier(estimators=[
        ('ada_boost', ada_clf), ('extra_trees', et_clf), ('gradient_boosting', gb_clf), ('log regg', lr_clf),
        ('perceptron', mlp_clf), ('random_forest', rf_clf), ('svm', svm_clf)], voting='soft'))

    voting_clf_scores1 = cross_val_score(voting_classifier1_pipeline,
                                         X_train, Y_train,
                                         cv=3,
                                         scoring='roc_auc',
                                         n_jobs=-1)

    voting_data1 = [
        {'mean_cv_scores': voting_clf_scores1.mean()}]

    voting_data1 = pd.DataFrame(voting_data1)
    voting_data1 = voting_data1.loc[voting_data1['mean_cv_scores'].idxmax()]
    voting_data1.to_csv('best_voting_classifier1.csv', index=False)

    voting_classifier1.fit(X_train_full, Y_train)
    filename = 'best_voting_classifier1.sav'
    pickle.dump(voting_classifier1, open(filename, 'wb'))

    voting_predict1 = voting_classifier1.predict_proba(X_test)
    voting1_roc_auc = roc_auc_score(Y_test, voting_predict1[:, 1])

    voting1_test_set = [
        {'roc_auc_score_on_test_set': voting1_roc_auc}]

    voting1_test_set = pd.DataFrame(voting1_test_set)
    voting1_test_set.to_csv('voting1_results_on_test_set.csv', index=False)

    return None

# Stacked classifiers on strong models
def run_stacked_classifier(X_train, X_test, Y_train, Y_test, X_train_full):
    # Load models
    ada_clf = pickle.load(open('best_ada_model.sav', 'rb'))
    et_clf = pickle.load(open('best_et_model.sav', 'rb'))
    gb_clf = pickle.load(open('best_gb_model.sav', 'rb'))
    lr_clf = pickle.load(open('best_logreg_model.sav', 'rb'))
    mlp_clf = pickle.load(open('best_mlp_model.sav', 'rb'))
    rf_clf = pickle.load(open('best_rf_model.sav', 'rb'))
    svm_clf = pickle.load(open('best_svm_model.sav', 'rb'))

    # Stacked Classifier 1
    print('running first stacked classifier using best models and gradient boosting as the meta classifier')
    sclf1 = StackingClassifier(classifiers=[ada_clf, et_clf, gb_clf, lr_clf, mlp_clf, rf_clf, svm_clf],
                               use_probas=True,
                               average_probas=False,
                               meta_classifier=GradientBoostingClassifier())

    sclf_pipeline1 = make_pipeline(MinMaxScaler(),
                                   StackingClassifier(classifiers=[ada_clf, et_clf, gb_clf, lr_clf, mlp_clf, rf_clf,
                                                                   svm_clf],
                                                      use_probas=True,
                                                      average_probas=False,
                                                      meta_classifier=GradientBoostingClassifier()))

    sclf_scores1 = cross_val_score(sclf_pipeline1, X_train, Y_train, cv=3, scoring='roc_auc', n_jobs=-1)

    sclf_data1 = [
        {'mean_cv_scores': sclf_scores1.mean()}]

    sclf_data1 = pd.DataFrame(sclf_data1)
    sclf_data1 = sclf_data1.loc[sclf_data1['mean_cv_scores'].idxmax()]
    sclf_data1.to_csv('best_stacked_classifier1.csv', index=False)

    sclf1.fit(X_train_full, Y_train)
    filename = 'best_stacked_classifier1.sav'
    pickle.dump(sclf1, open(filename, 'wb'))

    sclf_predict1 = sclf1.predict_proba(X_test)
    sclf1_roc_auc = roc_auc_score(Y_test, sclf_predict1[:, 1])

    sclf1_test_set = [
        {'roc_auc_score_on_test_set': sclf1_roc_auc}]

    sclf1_test_set = pd.DataFrame(sclf1_test_set)
    sclf1_test_set.to_csv('sclf1_results_on_test_set.csv', index=False)

    # Stacked Classifier 2
    print('running second stacked classifier using best models and logistic regression as the meta classifier')
    sclf2 = StackingClassifier(classifiers=[ada_clf, et_clf, gb_clf, lr_clf, mlp_clf, rf_clf, svm_clf],
                               use_probas=True,
                               average_probas=False,
                               meta_classifier=LogisticRegression())

    sclf_pipeline2 = make_pipeline(MinMaxScaler(),
                                   StackingClassifier(classifiers=[ada_clf, et_clf, gb_clf, lr_clf, mlp_clf, rf_clf,
                                                                   svm_clf],
                                                      use_probas=True,
                                                      average_probas=False,
                                                      meta_classifier=LogisticRegression()))

    sclf_scores2 = cross_val_score(sclf_pipeline2, X_train, Y_train, cv=3, scoring='roc_auc', n_jobs=-1)

    sclf_data2 = [
        {'mean_cv_scores': sclf_scores2.mean()}]

    sclf_data2 = pd.DataFrame(sclf_data2)
    sclf_data2 = sclf_data2.loc[sclf_data2['mean_cv_scores'].idxmax()]
    sclf_data2.to_csv('best_stacked_classifier2.csv', index=False)

    sclf2.fit(X_train_full, Y_train)
    filename = 'best_stacked_classifier2.sav'
    pickle.dump(sclf2, open(filename, 'wb'))

    sclf_predict2 = sclf2.predict_proba(X_test)
    sclf2_roc_auc = roc_auc_score(Y_test, sclf_predict2[:, 1])

    sclf2_test_set = [
        {'roc_auc_score_on_test_set': sclf2_roc_auc}]

    sclf2_test_set = pd.DataFrame(sclf2_test_set)
    sclf2_test_set.to_csv('sclf2_results_on_test_set.csv', index=False)

    # Stacked Classifier 3
    print('running third stacked classifier using vanilla models and gradient boosting as the meta classifier')
    sclf3 = StackingClassifier(classifiers=[AdaBoostClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(),
                                            LogisticRegression(), MLPClassifier(), RandomForestClassifier(),
                                            SVC(kernel='linear', probability=True)],
                               use_probas=True,
                               average_probas=False,
                               meta_classifier=GradientBoostingClassifier())

    sclf_pipeline3 = make_pipeline(MinMaxScaler(),
                                   StackingClassifier(classifiers=[AdaBoostClassifier(), ExtraTreesClassifier(),
                                            GradientBoostingClassifier(),
                                            LogisticRegression(), MLPClassifier(), RandomForestClassifier(),
                                            SVC(kernel='linear', probability=True)],
                                    use_probas=True,
                                    average_probas=False,
                                    meta_classifier=GradientBoostingClassifier()))

    sclf_scores3 = cross_val_score(sclf_pipeline3, X_train, Y_train, cv=3, scoring='roc_auc', n_jobs=-1)

    sclf_data3 = [
        {'mean_cv_scores': sclf_scores3.mean()}]

    sclf_data3 = pd.DataFrame(sclf_data3)
    sclf_data3 = sclf_data3.loc[sclf_data3['mean_cv_scores'].idxmax()]
    sclf_data3.to_csv('best_stacked_classifier3.csv', index=False)

    sclf3.fit(X_train_full, Y_train)
    filename = 'best_stacked_classifier3.sav'
    pickle.dump(sclf3, open(filename, 'wb'))

    sclf_predict3 = sclf3.predict_proba(X_test)
    sclf3_roc_auc = roc_auc_score(Y_test, sclf_predict3[:, 1])

    sclf3_test_set = [
        {'roc_auc_score_on_test_set': sclf3_roc_auc}]

    sclf3_test_set = pd.DataFrame(sclf3_test_set)
    sclf3_test_set.to_csv('sclf3_results_on_test_set.csv', index=False)

    # Stacked Classifier 4
    print('running fourth stacked classifier using vanilla models and logistic regression as the meta classifier')
    sclf4 = StackingClassifier(classifiers=[AdaBoostClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(),
                                            LogisticRegression(), MLPClassifier(), RandomForestClassifier(),
                                            SVC(kernel='linear', probability=True)],
                               use_probas=True,
                               average_probas=False,
                               meta_classifier=LogisticRegression())

    sclf_pipeline4 = make_pipeline(MinMaxScaler(),
                                   StackingClassifier(classifiers=[AdaBoostClassifier(), ExtraTreesClassifier(),
                                            GradientBoostingClassifier(),
                                            LogisticRegression(), MLPClassifier(), RandomForestClassifier(),
                                            SVC(kernel='linear', probability=True)],
                                                      use_probas=True,
                                                      average_probas=False,
                                                      meta_classifier=LogisticRegression()))

    sclf_scores4 = cross_val_score(sclf_pipeline4, X_train, Y_train, cv=4, scoring='roc_auc', n_jobs=-1)

    sclf_data4 = [
        {'mean_cv_scores': sclf_scores4.mean()}]

    sclf_data4 = pd.DataFrame(sclf_data4)
    sclf_data4 = sclf_data4.loc[sclf_data4['mean_cv_scores'].idxmax()]
    sclf_data4.to_csv('best_stacked_classifier4.csv', index=False)

    sclf4.fit(X_train_full, Y_train)
    filename = 'best_stacked_classifier4.sav'
    pickle.dump(sclf4, open(filename, 'wb'))

    sclf_predict4 = sclf4.predict_proba(X_test)
    sclf4_roc_auc = roc_auc_score(Y_test, sclf_predict4[:, 1])

    sclf4_test_set = [
        {'roc_auc_score_on_test_set': sclf4_roc_auc}]

    sclf4_test_set = pd.DataFrame(sclf4_test_set)
    sclf4_test_set.to_csv('sclf4_results_on_test_set.csv', index=False)

    # Stacked Classifier 5
    print('running fifth stacked classifier, using different models on different columns')

    segmented_pipe1 = make_pipeline(ColumnSelector(cols=(range(0, 19))),
                                    LogisticRegression())
    segmented_pipe2 = make_pipeline(ColumnSelector(cols=(range(19, 178))),
                                    GradientBoostingClassifier())

    sclf5 = StackingClassifier(classifiers=[segmented_pipe1, segmented_pipe2],
                               meta_classifier=RandomForestClassifier(n_estimators=100))

    sclf5_eval_pipe = make_pipeline(MinMaxScaler(), StackingClassifier(classifiers=[segmented_pipe1, segmented_pipe2],
                                                                       meta_classifier=RandomForestClassifier(
                                                                           n_estimators=100)))

    sclf5_search_best_cv = cross_val_score(sclf5_eval_pipe,
                                           X_train, Y_train,
                                           cv=5,
                                           scoring='roc_auc',
                                           n_jobs=-1)

    sclf5_data = [
        {'mean_cv_scores': sclf5_search_best_cv.mean()}]

    sclf5_model_results_df = pd.DataFrame(sclf5_data)
    best_sclf5 = sclf5_model_results_df.loc[sclf5_model_results_df['mean_cv_scores'].idxmax()]
    best_sclf5.to_csv('best_stacked_classifier5.csv', index=False)

    sclf5.fit(X_train_full, Y_train)
    filename = 'best_stacked_classifier5_model.sav'
    pickle.dump(sclf5, open(filename, 'wb'))

    sclf5_predict = sclf5.predict_proba(X_test)
    sclf5_roc_auc = roc_auc_score(Y_test, sclf5_predict[:, 1])

    sclf5_test_set = [
        {'roc_auc_score_on_test_set': sclf5_roc_auc}]

    sclf5_test_set = pd.DataFrame(sclf5_test_set)
    sclf5_test_set.to_csv('sclf5_results_on_test_set.csv', index=False)


if __name__ == "__main__":
    # Data In
    hof, batting, master, pitching, fielding, awards, all_stars, postseason, \
    world_series_and_cs = hall_of_fame_ingestion_and_wrangling()

    mitchell_players = run_webscraper_for_mitchell_report()
    suspended_players = run_webscraper_for_suspended_players()

    # Helpers
    df, df2 = clean_data(hof, batting, master, pitching, fielding, postseason, awards, all_stars, world_series_and_cs)
    df, df2 = add_players_connected_with_steroids(df, df2, mitchell_players, suspended_players)
    X_train, X_test, Y_train, Y_test = machine_learning_prep_classification(df)
    X_train_full = create_scaled_copy_of_X_train(X_train)

    # Classification
    run_voting_classifier(X_train, X_test, Y_train, Y_test, X_train_full)
    run_stacked_classifier(X_train, X_test, Y_train, Y_test, X_train_full)
