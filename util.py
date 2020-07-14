########################################################################
#
#   utilities (general) lib for SelfAdaptionUserProfiles
#
# ########################################################################

########################################################################
#   Imports
########################################################################
from scipy.stats import shapiro
from scipy.stats.stats import pearsonr
from scipy import stats
import random
from statsmodels.tools.eval_measures import rmse
#import operator
from numpy import matrix

import pandas as pd
import configuration



import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


########################################################################
#   Functions
########################################################################

def check_correlation(datafile_original, dimensions):
    print("checking correlations")
    dataset = datafile_original.copy()

    # build correlation matrix to detect possible relationships between the dimensions

    # check if data is normally distributed
    normdist = pd.DataFrame(0, index=dimensions, columns=['p-value normal distribution shapiro test'])
    print("checking if data is normally distributed")
    for dim in dimensions:
        if dim != 'sex':
            data = dataset[dim]
            stat, p = shapiro(data)
            print('%s : Statistics=%.3f, p=%.3f' % (dim,stat, p))
            normdist.loc[dim, 'p-value normal distribution shapiro test'] = round(p, 2)
    
    normdist.to_csv('normdist.csv', index=True, header=True, sep=',')

    # data is not normally distributed 
    # so we can not use pearson
    # let's try Spearman instead

    # create matrix
    print("calculating correlations and significants")
    matrix = pd.DataFrame(0, index=dimensions, columns=dimensions)
    matrix_sign = pd.DataFrame(0, index=dimensions, columns=dimensions)

    correlations = pd.DataFrame(0, index=dimensions, columns=['cor'])
    correlations['cor'] = ''
    correlations['cor'] = correlations['cor'].apply(list)

    for dimx in sorted(dimensions):
        for dimy in sorted(dimensions):
            #cor = round(pearsonr(dataset[dimx], dataset[dimy])[0], 2) 
            #sig = round(pearsonr(dataset[dimx], dataset[dimy])[1], 2)

            cor = round(stats.spearmanr(dataset[dimx], dataset[dimy])[0], 2) 
            sig = round(stats.spearmanr(dataset[dimx], dataset[dimy])[1], 2)
            matrix.loc[dimx, dimy] = cor
            matrix_sign.loc[dimx, dimy] = sig
            #print(dimx, " : ", dimy, " = ", cor)

            # lets save the most correlated dimensions for later use 
            if abs(cor) >= configuration.corr_min and sig <= configuration.corr_sign_max and "y_" in dimy and not "y_" in dimx:
                print("found potential correlated for dimension", dimy, " : ", dimx, "with ", cor, sig)
                correlations.loc[dimy, 'cor'].append(dimx)

    print("writing results to files")
    matrix.to_csv('matrix_correlation.csv', index=True, header=True, sep=',')
    matrix_sign.to_csv('matrix_correlation_sign.csv', index=True, header=True, sep=',')

    return correlations

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def display_result(label, res):
    print("########################################################################")
    print(label)
    print(res[0])

    print("rsquared             = ", res[1])
    print("rsquared_adj         = ", res[2])
    print("fitness_norm         = ", res[3])
    print("fitnes absolute      = ", res[4])
    print("compared with random = ", res[8])
    print("custom metric        = ", res[9])
    print("custom metric random = ", res[10])
    #print('model = ')
    #print(res[5])
    print("########################################################################")


def compare_intvl(y_real, y_pred):
    #print(y_real)
    #print(y_pred)
    count = 0
    for res, real in zip(y_pred, y_real):
        if res == real:
            count = count + 1
    #print((count / len(y_real) * 100, 0))
    return (count / len(y_real) * 100, 0)

def map_result(df):
    new_list = []
    for item in df:
        if item >= configuration.map_threshold:
            new_list.append(1)
        else:
            new_list.append(0)

    return new_list

def map_result3(df):
    new_list = []
    for item in df:
        if item >= 0.66:
            new_list.append(2)
        elif item >= 0.33:
            new_list.append(1)
        else:
            new_list.append(0)

    return new_list

def convert_result(r):
    if r >= configuration.map_threshold:
        return 1
    else:
        return 0

def compare(y_real, y_pred):
    # compare real values with predicted values:
    #r = rmse(y_real, y_pred)

    # normalize
    #rmse_norm = round(r / (max(y_real) - min(y_real)), 3)

    ## cm
    #randomlist = random.sample(range(1, 100), len(y_pred))
    #rmse_random = rmse(y_real, randomlist)

    # my own metric
    ret = [abs(m - n) for m,n in zip(y_real, y_pred)]
    

    #print(y_real, y_pred)
    #exit(0)
    
    count1 = 0
    count2 = 0
    for res in ret:
        if res <= configuration.metric_low:
            count1 = count1 + 1
        elif res <= configuration.metric_medium:
            count2 = count2 + 1

    return (round(count1/len(y_pred), 3), round(count2/len(y_pred), 3))



def lin_reg_plot(x, y, xlabel, ylabel, colors):
    # linear regression fitting
    print("_____")

    print(x.shape)
    print("_____")
    print("_____")
    print(y.shape)
    print("_____")


    lin_reg = LinearRegression()
    lin_reg.fit(x.values.reshape(-1, 1), y)

    # pearson cannot be used because data is not normally distributed
    #print(xlabel + ':' + ylabel, stats.spearmanr(xsingle, y))
    
    def viz_linear():
        plt.scatter(x, y, c=colors)
        plt.plot(x.values.reshape(-1, 1), lin_reg.predict(x.values.reshape(-1, 1)), color='blue')
        plt.title(xlabel + ':' + ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        return
    viz_linear()

def getColor(val):
    print(val)
    return 1