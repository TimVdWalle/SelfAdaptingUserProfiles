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
    correlations = util.check_correlation(dataset, configuration.dimensions_pol)
    print("correlations:", correlations)

    # generate linear models and calculate their metrics
    generate_polynomial_models(correlations, dataset)


def generate_polynomial_models(correlations, dataset):
    print("generating polynomial models")

    # loop all dependent variables that we want to build models for
    for dep in configuration.dependent:
        print("################################################################################################################################################")
        print("generating models for: ", dep)
        # create all combinations of independent variables that are potentially correlated for this dimension (dependent)
        loop_all_combinations_for(dep, correlations.loc[dep], dataset)
    
def loop_all_combinations_for(dep, correlations, dataset):
    independents = []
    independents_all = []
    best_fit = ('', 0, 0, 9999, 9999, '', [0], [0], 0)

    # first put all possible degree combinations in list ...
    for indep in correlations[0]:
        independents_all.append(indep)
        independents.append(indep)
        if not util.hasNumbers(indep):
            for degree in range(configuration.pol_min_degree, configuration.pol_max_degree + 1):
                independents_all.append(indep + " ** " + str(degree))
        
    #print(independents)
    #print(independents_all)
    
    # ... and now create every combination
    #print("independents_all = ", independents_all)
    for length in range(1, min(len(independents_all) + 1, configuration.max_dept_poly +1)):
        print("    generating length ", length)
        for subset in itertools.combinations(independents_all, length): 
            #print(subset)
            new = regression(dep, subset, dataset, independents)
            util.display_result(dep, new)

            if(new[4] < best_fit[4]):
                best_fit = new
                print("subset = ", subset)
                util.display_result(dep, new)
    
    util.display_result(dep + " best fit", best_fit)






def regression(dependent_str, model_list, dataset, independents_filter):
    # first we create the model for this dependent variable with the entire dataset
    independents_str = " + ".join(model_list)      
    print(independents_str)
    # https://stackoverflow.com/questions/48522609/how-to-retrieve-model-estimates-from-statsmodels

    X = dataset[sorted(independents_filter)]
    y = dataset[dependent_str]
    model = smf.ols(formula=dependent_str + " ~ " + independents_str, data=dataset).fit()

    # then we calculate the average fitness (rsme normalized) using k fold cross validation
    kf = KFold(configuration.kfold, True, 1)
    #print("########################################################################")
    fitness_norm = 0
    fitness = 0
    for train, test in kf.split(dataset):    
        model_t = smf.ols(formula=dependent_str + " ~ " + independents_str , data=dataset.iloc[train]).fit()

        # filter columns
        X = dataset[sorted(independents_filter)]
        y = dataset[dependent_str]

        # filter rows
        X = X.iloc[test]
        y = y.iloc[test]

        # generate predictions and metric
        ypred = model_t.predict(X)
        r = rmse(y, ypred)
        rmse_norm = round(r / (max(y) - min(y)), 3)
        #print("rmse_norm = ", rmse_norm)
        fitness_norm = fitness_norm + rmse_norm
        fitness = fitness + r
        
        # to be able to check manually
        #print(y)
        #print(ypred)


    fitness_norm = round(fitness_norm / configuration.kfold, 3)
    fitness = round(fitness / configuration.kfold, 3)
    #print("########################################################################")
    #print(fitness_norm, fitness)
    #print("########################################################################")

    rsquared = round(model.rsquared, 3)
    rsquared_adj = round(model.rsquared_adj, 3)

    X = dataset[sorted(independents_filter)]
    model_y = dataset[dependent_str]
    #model_y_pred = model_t.predict(X)

    # compare with random values
    #df_random = pd.DataFrame(np.random.randint(1,100,size=(len(model_y), 1)))
    randomlist = random.sample(range(1, 100), len(model_y))
    rmse_random = rmse(model_y, randomlist)
    
    #print("########################################################################")
    #print(model_y)
    #print(model_y_pred)
    #print("########################################################################")
    #print(model_y)
    #print(randomlist)
    #print("########################################################################")
    #print("rmse_random", rmse_random)

    #return (dep + " ~ " + independents, rsquared, rsquared_adj, fitness_norm, fitness, model.summary(), model_y, model_y_pred)
    return (dependent_str + " ~ " + independents_str, rsquared, rsquared_adj, fitness_norm, fitness, model.summary(), 0, 0, rmse_random)

