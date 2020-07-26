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

results = pd.DataFrame(index=['scie', 'math', 'sport', 'ent', 'hist', 'geo', 'arch',     'extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness'], columns=range(1,100))
results = results.fillna(0.0) # with 0s rather than NaNs

def run():
    print("running contentbased algorithm")

    # prepare data
    util_file.prepare_file(configuration.data_file)
    
    # read prepared data
    dataset = pd.read_csv(configuration.data_file_cleaned)

    for threshold in range(1,100):
        calculateAllProfiles(dataset, threshold)

    print(results)
    results.to_csv("content_based_results.csv")

def calculateAllProfiles(dataset, threshold):
    
    for index, profile in dataset.iterrows():
        #validationAll = validationAll + calculateProfile(profile, threshold)
        calculateProfile(profile, threshold)
    
    
    
    
    #print("validation of all profiles for threshold", threshold, ":", validationAll / dataset.shape[0])

def calculateProfile(profile, threshold):
    print("calculating profile ", profile['profileId'], profile['categories'])

    # CategoryScore = (CategoryRating * 100) / TotalRating

    # totalrating
    totalrating = 0
    for question in configuration.questions:
        if profile[question] == True:
            totalrating = totalrating + 1

    for feature in configuration.contentbased_interests:
        numberCorrect = 0
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

        r = numberCorrect / 80
        #print(feature[0], threshold, r)


        results[threshold][feature[0]] = results[threshold][feature[0]] + r

    for feature in configuration.contentbased_psy:
        #print("calculateing ", feature[0])
        numberCorrect = 0
        categoryRating = 0
        for featurequestion in feature[1]:
            if profile[featurequestion] == True:
                categoryRating = categoryRating + 1
        
        categoryScore = round(categoryRating * 100 / totalrating, 1)
        #print(categoryScore)
        #print(feature[0])
        if categoryScore >= threshold:
            #print(" ", feature[0], ":", categoryScore)
            #    y_intvl_openness_2
            #print("checking", 'y_intvl_' + feature[0] + '_2', " = ", profile['y_intvl_' + feature[0] + '_2'])
            if profile['y_intvl_' + feature[0] + '_2'] == 1:
                #print("corret 1")
                numberCorrect = numberCorrect + 1

        else:
            if profile['y_intvl_' + feature[0] + "_2"] == 0:
                #print("corret 2")
                numberCorrect = numberCorrect + 1

        r = numberCorrect / 80
        #results


        results[threshold][feature[0]] = results[threshold][feature[0]] + r        #print(feature[0], threshold, r)

        
    #return numberCorrect/len(configuration.contentbased)