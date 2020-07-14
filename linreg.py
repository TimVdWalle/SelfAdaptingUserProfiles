########################################################################
#
#   linear regression lib for SelfAdaptionUserProfiles
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
    print("correlations:", correlations)

    # generate linear models and calculate their metrics
    generate_linear_models(correlations, dataset)


def generate_linear_models(correlations, dataset):
    print("generating linear models")

    # loop all dependent variables that we want to build models for
    for dep in configuration.dependent:
        print("################################################################################################################################################")
        print("generating models for: ", dep)
        # create all combinations of independent variables that are potentially correlated for this dimension (dependent)
        loop_all_combinations_for(dep, correlations.loc[dep], dataset)
    

def loop_all_combinations_for(dep, correlations, dataset):
    # initialize vars with dummies and sentinels
    highest = ('', 0, 0, 9999, 9999, '', [0], [0], 0, 0, 0)
    highest_adj = ('', 0, 0, 9999, 9999, '', [0], [0], 0, 0, 0)
    best_fit = ('', 0, 0, 9999, 9999, '', [0], [0], 0, 0, 0)

    # first generate model with all independent variables
    # generating model with all independents
    new = regression(dep, configuration.independent, dataset)
    highest = new
    highest_adj = new
    best_fit = new
    util.display_result("model with all independents", new)

    # second:  generate model with only the major independent variables
    # generating model with all independents
    new = regression(dep, configuration.independent_major, dataset)
    util.display_result("model with only major independents", new)

    print("creating all combinations for ", dep, "of :")
    print(correlations[0])

    # generate all combinations for the dimensions that are potentially correlation for dimension <dep>
    for length in range(1, min(len(correlations[0]) + 1, configuration.max_depth +1)):
        print("    generating length ", length)
        for subset in itertools.combinations(correlations[0], length): 
            independents = " + ".join(subset)         
            #print("    ", dep, " => ", independents)
            
            # new contains metrics as a tuple
            # regression generates models and searches for best fit, highest rsquared, and rsquared_adjusted
            new = regression(dep, subset, dataset)
            if(new[1] > highest[1]):
                highest = new

            if(new[2] > highest_adj[2]):
                highest_adj = new

            if(new[9][0] > best_fit[9][0]):
                best_fit = new
    
    util.display_result(dep + " highest squared", highest)
    util.display_result(dep + " highest squared_adj", highest_adj)
    util.display_result(dep + " best fit", best_fit)
   
    #pd.DataFrame(best_fit[6]).to_csv(dep + "_y.csv", index=False, encoding='utf8')
    #pd.DataFrame(best_fit[7]).to_csv(dep + "_ypred.csv", index=False, encoding='utf8')



def regression(dep, subset, dataset):
    # first we create the model for this dependent variable with the entire dataset
    independents = " + ".join(subset)      
    # https://stackoverflow.com/questions/48522609/how-to-retrieve-model-estimates-from-statsmodels
    model = smf.ols(formula=dep + " ~ " + independents , data=dataset).fit()

    # then we calculate the average fitness (rsme normalized) using k fold cross validation
    kf = KFold(configuration.kfold, True, 1)
    #print("########################################################################")
    fitness_norm = 0
    fitness = 0
    compared = (0,0)
    for train, test in kf.split(dataset):    
        model_t = smf.ols(formula=dep + " ~ " + independents , data=dataset.iloc[train]).fit()
        #print(model_t.summary())

        # filter columns
        X = dataset[sorted(subset)]
        y = dataset[dep]

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
        
        ypred_mapped = util.map_result(ypred)

        #print(ypred)
        #print(ypred_mapped)
        
        result_compared = util.compare_intvl(y, ypred_mapped)

        compared = (compared[0] + result_compared[0], compared[1] + result_compared[1])
        

        # to be able to check manually
        #print(y)
        #print(ypred)


    fitness_norm = round(fitness_norm / configuration.kfold, 3)
    fitness = round(fitness / configuration.kfold, 3)
    compared = (round(compared[0] / configuration.kfold, 3), round(compared[1] / configuration.kfold, 3))

    #print("########################################################################")
    #print(fitness_norm, fitness)
    #print("########################################################################")

    rsquared = round(model.rsquared, 3)
    rsquared_adj = round(model.rsquared_adj, 3)

    X = dataset[sorted(subset)]
    model_y = dataset[dep]
    #model_y_pred = model_t.predict(X)

    # compare with random values
    #df_random = pd.DataFrame(np.random.randint(1,100,size=(len(model_y), 1)))
    #randomlist = random.sample(range(1, 100), len(model_y))

    randomlist = [random.randint(0,1) for x in range(len(model_y))]
    rmse_random = rmse(model_y, randomlist)

    #compared_random = util.compare_intvl(model_y, randomlist)
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
    return (dep + " ~ " + independents, rsquared, rsquared_adj, fitness_norm, fitness, model.summary(), 0, 0, rmse_random, compared, compared_random)

