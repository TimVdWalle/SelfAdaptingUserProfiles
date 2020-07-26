########################################################################
#
#   Self Adapting User Profiles
#   Tim Vande Walle
#   2019-2020
#   Thesis VUB
#   promotor: Olga De Troyer
#
########################################################################

########################################################################
#
#   linear regression lib for SelfAdaptionUserProfiles
#   https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
#
# ########################################################################

########################################################################
#   Imports
########################################################################
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import configuration
import util
import util_file


########################################################################
#   Functions General
########################################################################
def runParTuning():
    # prepare data
    util_file.prepare_file(configuration.data_file)
    
    # read prepared data
    dataset = pd.read_csv(configuration.data_file_cleaned)


    for dep in configuration.dependent:
        print("running parameter tuning maxdepth for :", dep)
        partun_max_depth(dataset, dep)

    # parameter tuning
    for dep in configuration.dependent:
        print("running parameter tuning n_estimators for :", dep)
        partun_n_estimators(dataset, dep)
    
    #partun_max_depth(dataset, 'y_intvl_openness')
    

def run():
    # prepare data
    util_file.prepare_file(configuration.data_file)
    
    # read prepared data
    dataset = pd.read_csv(configuration.data_file_cleaned)

    # creating random forest
    create_random_forests(dataset)

def split_data(dep, dataset):
    # splitting data
    data_train, data_test = train_test_split(dataset, test_size=configuration.rf_test_size)

    x_train = data_train[configuration.independent]
    y_train = data_train[dep]

    x_test = data_test[configuration.independent]
    y_test = data_test[dep]

    return x_train, y_train, x_test, y_test


########################################################################
#   Functions Parameter Tuning
########################################################################
def partun_n_estimators(dataset, dep):
    print("parameter tuning n_estimators")

    train_results = []
    test_results = []
    train_scores = []
    test_scores = []

    for estimator in configuration.rf_pt_n_estimators:
        print("trying n_estimator", estimator)
        rmse_train = 0
        rmse_test  = 0
        score_train = 0
        score_test = 0

        for l in range(1, configuration.rf_pt_n_estimators_loop +1):
            # splitting data
            x_train, y_train, x_test, y_test = split_data(dep, dataset)

            rf = RandomForestClassifier(n_estimators=estimator, max_depth=configuration.rf_max_depth, n_jobs=-1)
            rf.fit(x_train, y_train)
            train_pred = rf.predict(x_train)            
            rmse_train = rmse_train + rmse(y_train, train_pred)      
            score_train = score_train + rf.score(x_train, y_train)      
                        
            y_pred = rf.predict(x_test)
            rmse_test = rmse_test + rmse(y_test, y_pred)
            score_test = score_test + rf.score(x_test, y_test)
            
        train_results.append(rmse_train / configuration.rf_pt_n_estimators_loop)
        test_results.append(rmse_test / configuration.rf_pt_n_estimators_loop)

        train_scores.append(score_train / configuration.rf_pt_n_estimators_loop * 100)
        test_scores.append(score_test / configuration.rf_pt_n_estimators_loop * 100)


    #line1, = plt.plot(configuration.rf_pt_n_estimators, train_results, 'b', label="Train rmse")
    #line2, = plt.plot(configuration.rf_pt_n_estimators, test_results, 'r', label="Test rmse")

    line3, = plt.plot(configuration.rf_pt_n_estimators, train_scores, 'g', label="Train score")
    line4, = plt.plot(configuration.rf_pt_n_estimators, test_scores, 'y', label="Test score")

    plt.legend(handler_map={line3: HandlerLine2D(numpoints=2)})
    plt.ylabel('score')
    plt.xlabel('n_estimators')
    plt.title('n_estimators parameter tuning for ' + dep)
    plt.show()

def partun_max_depth(dataset, dep):
    print("parameter tuning max_depth")

    train_results = []
    test_results = []
    train_scores = []
    test_scores = []

    for max_depth in configuration.rf_pt_max_depths:
        print("trying max_depth", max_depth)
        rmse_train = 0
        rmse_test  = 0
        score_train = 0
        score_test = 0

        for l in range(1, configuration.rf_pt_max_depth_loop +1):
            # splitting data
            x_train, y_train, x_test, y_test = split_data(dep, dataset)

            rf = RandomForestClassifier(n_estimators=configuration.rf_n_estimators, max_depth=max_depth, n_jobs=-1)
            rf.fit(x_train, y_train)
            train_pred = rf.predict(x_train)            
            rmse_train = rmse_train + rmse(y_train, train_pred)
            score_train = score_train + rf.score(x_train, y_train)

            y_pred = rf.predict(x_test)
            rmse_test = rmse_test + rmse(y_test, y_pred)
            score_test = score_test + rf.score(x_test, y_test)
            
        #train_results.append(rmse_train / configuration.rf_pt_max_depth_loop)
        #test_results.append(rmse_test / configuration.rf_pt_max_depth_loop)

        train_scores.append(score_train / configuration.rf_pt_max_depth_loop * 100)
        test_scores.append(score_test / configuration.rf_pt_max_depth_loop * 100)

    #line1, = plt.plot(configuration.rf_pt_max_depths, train_results, 'b', label="Train rmse")
    #line2, = plt.plot(configuration.rf_pt_max_depths, test_results, 'r', label="Test rmse")

    line3, = plt.plot(configuration.rf_pt_max_depths, train_scores, 'g', label="Train score")
    line4, = plt.plot(configuration.rf_pt_max_depths, test_scores, 'y', label="Test score")

    plt.legend(handler_map={line3: HandlerLine2D(numpoints=2)})
    plt.ylabel('score')
    plt.xlabel('tree depth')
    plt.title('max_depth parameter tuning for ' + dep)
    plt.show()


########################################################################
#   Functions to create the Random Forest
########################################################################
def create_random_forests(data_file):
    for dep in configuration.dependent:
        print("creating random forest for ", dep)
        create_random_forest(dep, data_file)


def create_random_forest(dep, dataset):
    compared = 0
    compared_random = 0
    r = 0
    rmse_random = 0

    for x in range(0,configuration.rf_loops):
        # splitting data
        x_train, y_train, x_test, y_test = split_data(dep, dataset)

        #x_test = x_train
        #y_test = y_train

        rf = RandomForestClassifier(n_estimators=configuration.rf_n_estimators, max_depth=configuration.rf_max_depth)
        rf.fit(x_train, y_train)
        
        # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        # max_depth=None, max_features=’auto’, max_leaf_nodes=None,
        # min_impurity_split=1e-07, min_samples_leaf=1,
        # min_samples_split=2, min_weight_fraction_leaf=0.0,
        # n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
        # verbose=0, warm_start=False)
        y_pred = rf.predict(x_test)

        # print(y_test)
        # print("_____________")
        # print(y_pred)

        r = r + rmse(y_test, y_pred)
        #print(r)

        score = rf.score(x_test, y_test)
        #print(score)


        randomlist = [random.randint(0,1) for x in range(len(y_test))]
        #randomlist = [random.randint(0,100) for x in range(len(y_test))]

        rmse_random = rmse_random + rmse(y_test, randomlist)

        compared = compared + util.compare_intvl(y_test, y_pred)[0]                   # for interval dependent variables
        compared_random = compared_random + util.compare_intvl(y_test, randomlist)[0]        # for interval dependent variables
        #compared_random = util.compare(model_y, randomlist)            # for percentile dependent variables


    print("accuracy=", compared / configuration.rf_loops)
    print("accuracy random=", compared_random / configuration.rf_loops)
    print("rmse=", r / configuration.rf_loops)
    print("rmse random=", rmse_random / configuration.rf_loops)
    print("################################################################################################################################")