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
#   polynomial regression lib for SelfAdaptionUserProfiles
#
# ########################################################################

########################################################################
#   Imports
########################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random

from scipy.stats.stats import pearsonr
from scipy import stats

from numpy.random import seed
from numpy.random import randn
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 



import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

import configuration
import util
import util_file



########################################################################
#   Functions
########################################################################
def run():
    # prepare data
    util_file.prepare_file(configuration.data_file)
    
    # read prepared data
    dataset = pd.read_csv(configuration.data_file_cleaned)

    # check correlation
    correlations = util.check_correlation(dataset, configuration.dimensions)
    #print("correlations:", correlations)

    # generate linear models and calculate their metrics
    generate_models(correlations, dataset)


def generate_models(correlations, dataset):
    print("generating logistic regression models - with pca + feature selection")

    # loop all dependent variables that we want to build models for
    for dep in configuration.dependent:
        print("################################################################################################################################################")
        print("generating models for: ", dep)
        # create all combinations of independent variables that are potentially correlated for this dimension (dependent)
        loop_all_combinations_for(dep, correlations.loc[dep], dataset)
    
def loop_all_combinations_for(dep, correlations, dataset):
    best_fit = ('', 0, 0, 9999, 9999, '', [0], [0], 0, (0,0), 0)
    highest  = ('', 0, 0, 9999, 9999, '', [0], [0], 0, 0, 0)

    for length in range(2, min(len(correlations[0]) + 1, configuration.max_dept_logreg +1)):
        print("    generating length ", length)
        for subset in itertools.combinations(correlations[0], length):
            for pca_n in range(configuration.pca_min_n, min(configuration.pca_max_n + 1, len(subset)+1)): 
                #print("    generating pca depth ", pca_n, subset)

                new = regression(dep, subset, dataset, pca_n)
                #util.display_result(dep + " current = ", new)

                #print(new[9][0])

                # saving if rsquared is better
                if(new[1] > highest[1]):
                    highest = new

                # saving if fitness is better
                # if(new[4] < best_fit[4]):
                #     best_fit = new

                # saving if fitness is better
                if(new[9][0] > best_fit[9][0]):
                    best_fit = new

    util.display_result(dep + " highest squared", highest)
    util.display_result(dep + " best fit", best_fit)

    #exit(0)


def regression(dep, subset, dataset, pca_n):
    #print(subset)
    #print(pca_n)

    X = dataset[sorted(subset)]
    y = dataset[dep]

    randint = random.randint(1, 1000)
    kf = KFold(n_splits=configuration.kfold, shuffle=True, random_state=randint)
    fitness_norm = 0
    fitness = 0
    compared = (0,0)
    for train, test in kf.split(dataset):
        X_train = X.iloc[train]
        y_train = y.iloc[train]

        X_test = X.iloc[test]
        y_test = y.iloc[test]

        scaler = StandardScaler()
        scaler.fit(X_train)

        x_train_scaled = scaler.transform(X_train)
        x_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=pca_n)
        pca.fit(x_train_scaled)

        x_train_pca = pca.transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)
   
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
        model = mul_lr.fit(x_train_pca, y_train)

        ypred = model.predict(x_test_pca)

        r = rmse(y_test, ypred)
        #rmse_norm = round(r / (max(y_test) - min(y_test)), 3)
        rmse_norm = 0
        
        fitness_norm = fitness_norm + rmse_norm
        fitness = fitness + r
        
        result_compared = util.compare_intvl(y_test, ypred)
        #result_compared = util.compare(y_test, ypred)

        compared = (compared[0] + result_compared[0], compared[1] + result_compared[1])

    fitness_norm = round(fitness_norm / configuration.kfold, 3)
    fitness = round(fitness / configuration.kfold, 3)
    compared = (round(compared[0] / configuration.kfold, 3), round(compared[1] / configuration.kfold, 3))
    #print("########################################################################")
    #print(fitness_norm, fitness)
    #print("########################################################################")

    scaler_all = StandardScaler()
    scaler_all.fit(X)

    x_scaled = scaler_all.transform(X)

    pca_all = PCA(n_components=pca_n)
    pca.fit(x_scaled)
    x_pca = pca.transform(x_scaled)

    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model = mul_lr.fit(x_pca, y)

    #ypred = model.predict(x_test_pca)




    rsquared = model.score(x_pca, y)
    #rsquared = -1
    rsquared_adj = -1

    model_y = dataset[dep]
    # randomlist = random.sample(range(1, 85), len(model_y))        # not usefull because it does not allow for duplicates

    randomlist = [random.randint(0,1) for x in range(len(model_y))]
    #randomlist = [random.randint(0,100) for x in range(len(model_y))]

    rmse_random = rmse(model_y, randomlist)

    compared_random = util.compare_intvl(model_y, randomlist)      # for interval dependent variables
    #compared_random = util.compare(model_y, randomlist)             # for percentile dependent variables
    
    return ("pca_n:" + str(pca_n) + " = " + dep + " ~ " + "+".join(subset), rsquared, rsquared_adj, fitness_norm, fitness, '', 0, 0, rmse_random, compared, compared_random)
