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

"""
Inspired by the works of Gopidi (2015), we will use this formula when calculating the new values for a certain category C, where ratings can either be simple booleans or numbers on a scale (for instance between 0 and 10).
CategoryScore = (CategoryRating * 100) / TotalRating

Where 
CategoryRating 
= for a category C, we sum all weights/booleans from all content items that have a value for this category and that the user has interacted with

TotalRating 
= sum of all weights/booleans from all the categories from the content items that the user interacted with 
"""


def run():
    print("running contentbased algorithm")

    # prepare data
    util_file.prepare_file(configuration.data_file)
    
    # read prepared data
    dataset = pd.read_csv(configuration.data_file_cleaned)

    for threshold in range(1,99):
        calculateAllProfiles(dataset, threshold)

def calculateAllProfiles(dataset, threshold):
    validationAll = 0
    for index, profile in dataset.iterrows():
        validationAll = validationAll + calculateProfile(profile, threshold)
    
    print("validation of all profiles for threshold", threshold, ":", validationAll / dataset.shape[0])

def calculateProfile(profile, threshold):
    #print("calculating profile ", profile['profileId'], profile['categories'])

    # CategoryScore = (CategoryRating * 100) / TotalRating

    # totalrating
    totalrating = 0
    numberCorrect = 0
    for question in configuration.questions:
        if profile[question] == True:
            totalrating = totalrating + 1

    for feature in configuration.contentbased:

        categoryRating = 0
        for featurequestion in feature[1]:
            if profile[featurequestion] == True:
                categoryRating = categoryRating + 1
        
        categoryScore = round(categoryRating * 100 / totalrating, 1)
        if categoryScore >= threshold:
            #print(" ", feature[0], ":", categoryScore)
            if profile['y_' + feature[0]] == 1:
                numberCorrect = numberCorrect + 1

        else:
            if profile['y_' + feature[0]] == 0:
                numberCorrect = numberCorrect + 1
        
    return numberCorrect/len(configuration.contentbased)