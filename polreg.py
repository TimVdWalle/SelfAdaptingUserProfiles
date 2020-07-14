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
    correlations = util.check_correlation(dataset, configuration.dimensions)
    #print("correlations:", correlations)

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
    highest = ('', 0, 0, 9999, 9999, '', [0], [0], 0, (0,0), 0)
    best_fit = ('', 0, 0, 9999, 9999, '', [0], [0], 0, (0,0), 0)

    for length in range(2, min(len(correlations[0]) + 1, configuration.max_dept_poly +1)):
            print("    generating length ", length)
            for subset in itertools.combinations(correlations[0], length):
                for degree in range(configuration.pol_min_degree, configuration.pol_max_degree + 1):
                    new = regression(dep, subset, dataset, degree)
                    #util.display_result(dep, new)

                    # saving if rsquared is better
                    if(new[1] > highest[1]):
                        highest = new

                    # saving if fitness is better
                    if(new[9][0] > best_fit[9][0]):
                        best_fit = new

    util.display_result(dep + " highest squared", highest)
    util.display_result(dep + " best fit", best_fit)


# def dummy():
#     print(dep)
#     print(correlations)
#     print('done')



#     X = dataset[sorted(correlations[0])]
#     y = dataset[dep]
    
#     poly = PolynomialFeatures(degree=30)
#     poly_variables = poly.fit_transform(X)

#     #poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables, results, test_size = 0.3, random_state = 4)

#     regression = linear_model.LinearRegression()
#     model = regression.fit(poly_variables, y)
#     score = model.score(poly_variables, y)

#     print(model.get_params)
#     print(model.coef_)
#     print(score)


#     exit(0)

def regression(dep, subset, dataset, degree):
    X = dataset[sorted(subset)]
    y = dataset[dep]

    # first we create the model for this dependent variable with the entire dataset
    # https://stackoverflow.com/questions/48522609/how-to-retrieve-model-estimates-from-statsmodels
    
    poly = PolynomialFeatures(degree=degree)
    poly_variables_all = poly.fit_transform(X)

    regression = linear_model.LinearRegression()
    model = regression.fit(poly_variables_all, y)

    # then we calculate the average fitness (rsme normalized) using k fold cross validation
    kf = KFold(configuration.kfold, True, 1)
    #print("########################################################################")
    fitness_norm = 0
    fitness = 0
    compared = (0,0)
    for train, test in kf.split(dataset):
        # filter rows
        X_train = X.iloc[train]
        y_train = y.iloc[train]

        X_test = X.iloc[test]
        y_test = y.iloc[test]

        poly = PolynomialFeatures(degree=degree)
        poly_variables_train = poly.fit_transform(X_train)
        poly_variables_test = poly.fit_transform(X_test)
        

        regression_t = linear_model.LinearRegression()
        model_t = regression_t.fit(poly_variables_train, y_train)

        # generate predictions and metric
        ypred = model_t.predict(poly_variables_test)

        r = rmse(y_test, ypred)
        #rmse_norm = round(r / (max(y_test) - min(y_test)), 3)
        rmse_norm = -1
        
        fitness_norm = fitness_norm + rmse_norm
        fitness = fitness + r

        ypred_mapped = util.map_result3(ypred)
        
        #result_compared = util.compare_intvl(y_test, ypred)
        result_compared = util.compare_intvl(y_test, ypred_mapped)
        compared = (compared[0] + result_compared[0], compared[1] + result_compared[1])
        
    fitness_norm = round(fitness_norm / configuration.kfold, 3)
    fitness = round(fitness / configuration.kfold, 3)
    compared = (round(compared[0] / configuration.kfold, 3), round(compared[1] / configuration.kfold, 3))
    #print("########################################################################")
    #print(fitness_norm, fitness)
    #print("########################################################################")

    rsquared = model.score(poly_variables_all, y)
    rsquared_adj = -1

    #X = dataset[sorted(independents_filter)]
    #model_y = dataset[dependent_str]
    #model_y_pred = model_t.predict(X)

    # compare with random values
    #df_random = pd.DataFrame(np.random.randint(1,100,size=(len(model_y), 1)))
    #randomlist = random.sample(range(1, 100), len(model_y))
    #rmse_random = rmse(model_y, randomlist)

    model_y = dataset[dep]
    #randomlist = random.sample(range(0,1), len(model_y))    
    randomlist = [random.randint(0,2) for x in range(len(model_y))]

    rmse_random = rmse(model_y, randomlist)

    compared_random = util.compare_intvl(model_y, randomlist)
    
    #print("########################################################################")
    #print(model_y)
    #print(model_y_pred)
    #print("########################################################################")
    #print(model_y)
    #print(randomlist)
    #print("########################################################################")
    #print("rmse_random", rmse_random)

    #return (dep + " ~ " + independents, rsquared, rsquared_adj, fitness_norm, fitness, model.summary(), model_y, model_y_pred)
    return ("degree:" + str(degree) + " = " + dep + " ~ " + "+".join(subset), rsquared, rsquared_adj, fitness_norm, fitness, '', 0, 0, rmse_random, compared, compared_random)



    #return (dep + " ~ " + independents, rsquared, rsquared_adj, fitness_norm, fitness, model.summary(), 0, 0, rmse_random, compared, compared_random)
