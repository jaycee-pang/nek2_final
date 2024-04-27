import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
import shutil
import matplotlib 
from scipy.spatial.distance import euclidean

from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import KFold

import imblearn as imb
from sklearn.metrics import confusion_matrix
import itertools

from scipy.stats import randint, zscore
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import sys
sys.path.append('../')
from sklearn.model_selection import GridSearchCV



def calculate_metrics(y_true, y_pred): 
    """Counts of tp, tn, fp, fn"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def rf_results(model, x_input, true_labels): 
    """predictions and get metrics 
    @params
        - model: fitted model (fitted to train set)
        - x_input (np.array): x feats (either train or test)
        - true_labels (np.array): corresponding true labels 
    @returns
    dictionary of results and metrics"""
    predictions = model.predict(x_input)
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    tp, tn, fp, fn = calculate_metrics(true_labels, predictions)
    specificity = tn / (tn + fp)
    probability = model.predict_proba(x_input)
    print(f'accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, specificity: {specificity:.3f}')

    return {'predictions':predictions, 'accuracy':acc , 'precision': precision, 'recall':recall, 'specificity':specificity}

def rf_models(train_x, train_y, test_x, test_y, rf_type, parameters):
    """Fit a RF model, make predictions, get probabilities
    @params: 
        - train_x, train_y, test_x, test_y: train and test set inputs (np arrays) 
        - rf_type (str): model type: RandomForestClassifier, RandomForestClassifier with class_weight:'balanced', or BalancedRandomForestClassifier
        default is RFC 
        - parameters: dict for model params 

    @returns: dict with model, train/test prections and probabilities
    """
    n_estimators = parameters.get('n_estimators', 100)
    random_state = parameters.get('random_state', 42) 
    criterion = parameters.get('criterion', 'gini')
    max_depth = parameters.get('max_depth', 100)
    min_samples_split = parameters.get('min_samples_split', 2) 
    min_samples_leaf = parameters.get('min_samples_leaf', 1) 
    bootstrap = parameters.get('bootstrap', False) 
    max_features = parameters.get('max_features', None) 
    class_weight = parameters.get('class_weight', None)
    
    if (rf_type == 'balanced class_weight'): 
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')
    elif (rf_type == 'balanced RF'):
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    

    model.fit(train_x, train_y)
    model_results = rf_results2(model, train_x, train_y, test_x, test_y)

    classes = ['0','1']
    
    return {'model': model, 'train_pred':model_results['train_pred'], 'test_pred': model_results['test_pred'],
             'train_prob':model_results['train_prob'], 'test_prob': model_results['test_prob']}


def find_best_models(train_x, train_y, test_x, test_y, rf_type, parameters, param_dist, dataset_type,verbose_val=None):
    """uses GridSearchCV not random grid search
    Grid search to find the best model, make predictions (train and test), get probability (train and test), and plot CM 
    Save best model to pickle file 
    @params:
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays) 
    rf_type: model type: RandomForestClassifier, RandomForestClassifier with class_weight:'balanced', or BalancedRandomForestClassifier
        default is RFC 
    parameters: dict for model params 
    param_dist: parameters for grid search
    dataset_type: binding or inhibition
    @returns: dict with model, train/test prections and probabilities
    """
    n_estimators = parameters.get('n_estimators', 100)
    random_state = parameters.get('random_state', 42) 
    criterion = parameters.get('criterion', 'gini')
    max_depth = parameters.get('max_depth', 100)
    min_samples_split = parameters.get('min_samples_split', 2) 
    min_samples_leaf = parameters.get('min_samples_leaf', 1) 
    bootstrap = parameters.get('bootstrap', False) 
    max_features = parameters.get('max_features', None) 
    class_weight = parameters.get('class_weight', None)
    bootstrap = parameters.get('bootstrap', False)
    if (verbose_val==None): 
        verbose_val = 0
    if (rf_type == 'balanced class_weight'): 
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')
    elif (rf_type == 'balanced RF'):
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)

    rand_search = GridSearchCV(estimator =model, param_grid = param_dist,cv=5, n_jobs=8, verbose=verbose_val)
    rand_search.fit(train_x, train_y) 
    best_rf = rand_search.best_estimator_

    model_results = rf_results2(best_rf, train_x, train_y, test_x, test_y)
    classes = ['0','1']


    return {'best_model': best_rf, 'train_pred':model_results['train_pred'], 'test_pred': model_results['test_pred'],
             'train_prob':model_results['train_prob'], 'test_prob': model_results['test_prob']}





def rf_results2(model, train_x, train_y, test_x, test_y): 
    """Make predictions adn get probabilities (for train and test)
    @params
    model: fitted model (fitted to train set)
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays)
    @returns dict 
    train/test predictions
    train/test accuracies 
    train/test probabilities"""
    train_pred = model.predict(train_x) 
    test_pred = model.predict(test_x)
    train_acc = accuracy_score(train_y, train_pred) 
    test_acc = accuracy_score(test_y, test_pred) 
    
    precision_train = precision_score(train_y, train_pred)
    precision_test = precision_score(test_y, test_pred)

    recall_train = recall_score(train_y, train_pred)
    recall_test = recall_score(test_y, test_pred)

    tp_train, tn_train, fp_train, fn_train = calculate_metrics(train_y, train_pred)
    tp_test, tn_test, fp_test, fn_test = calculate_metrics(test_y, test_pred)
    sensitivity_train = tp_train / (tp_train  + fn_train)
    sensitivity_test = tp_test / (tp_test + fn_test)


    specificity_train = tn_train / (tn_train  + fp_train)
    specificity_test = tn_test / (tn_test + fp_test)

    train_prob = model.predict_proba(train_x) 
    test_prob = model.predict_proba(test_x) 

    print(f'TRAIN: accuracy: {train_acc:.3f}, precision: {precision_train:.3f}, recall: {recall_train:.3f}, specificity: {specificity_train:.3f}')
    print(f'TEST: accuracy: {test_acc:.3f}, precision: {precision_test:.3f}, recall: {recall_test:.3f}, specificity: {specificity_test:.3f}')

    

    return {'train_pred':train_pred, 'test_pred':test_pred,
            'train_acc':train_acc, 'test_acc':test_acc,
            'train_prob':train_prob, 'test_prob':test_prob, 
            'train_acc': train_acc, 'test_acc': test_acc,
            'train_prec':precision_train, 'test_prec': precision_test, 
            'train_recall': recall_train, 'test_recall': recall_test, 
            'train_specificity': specificity_train, 'test_specificity': specificity_test}

def save_rf_results(model, x_input, true_labels):
    """Save rf model results to DF"""
    results = rf_results(model, x_input, true_labels)
    results_df = pd.DataFrame(results)
    results_df['y'] = true_labels
    results_df['prob_class0'] = model.predict_proba(x_input)[:,0] 
    results_df['prob_class1'] = model.predict_proba(x_input)[:,1] 
    return results_df 

def calculate_indicies(y_pred, y_true): 
    tp = np.where((y_pred == 1) & (y_true == 1))[0]
    fp = np.where((y_pred == 1) & (y_true == 0))[0]
    tn = np.where((y_pred == 0) & (y_true == 0))[0]
    fn = np.where((y_pred == 0) & (y_true == 1))[0]
    

    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

