########################################################################
#
#   preprocessing and analysing for SelfAdaptionUserProfiles
#
# ########################################################################

########################################################################
#   Imports
########################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random as rd

from scipy.stats.stats import pearsonr
from scipy import stats

from numpy.random import seed
from numpy.random import randn
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from sklearn.decomposition import PCA
from sklearn import preprocessing

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
    #correlations = util.check_correlation(dataset, configuration.dimensions)
    #print("correlations:", correlations)

    #run_pca(dataset)
    pca_plots(dataset)

    # draw scatterplots
    #scatter_plots(dataset)

    # draw pca_plots
    #calculate_best_pca_plots(dataset)

def scatter_plots(dataset):
    print("scatterplots")
    
    plot_settings = [
        ("beta_norm",                   "y_scie",                       "beta (normalized)",                "science (preference)",         "y_scie"),
        ("math_norm",                   "y_math",                       "math (normalized)",                "math (preference)",            "y_math"),
        ("sport_norm",                  "y_sport",                      "sport (normalized)",               "sport (preference)",           "y_sport"),
        ("ent_norm",                    "y_ent",                        "entertainment (normalized)",       "entertainment (preference)",   "y_ent"),
        ("hist_norm",                   "y_hist",                       "history (normalized)",             "history (preference)",         "y_hist"),
        ("hist_norm",                   "y_geo",                        "history (normalized)",             "geography (preference)",       "y_geo"),
        ("arch_norm",                   "y_arch",                       "architecture (normalized)",        "architecture (preference)",         "y_arch"),


        ("hist_norm",                   "y_pctl_openness",              "history (normalized)",             "openness",             "y_intvl_openness"),
        ("total_categories_norm",       "y_pctl_conscientiousness",     "total categories (normalized)",    "conscientiousness",    "total_categories_norm"),
        ("age",                         "y_pctl_agreeableness",         "age",                              "agreeableness",        "y_intvl_agreeableness"),
        ("sex",                         "y_pctl_neuroticism",           "sex",                              "neuroticism",          "y_intvl_neuroticism"),
        ("do_question_group_norm",      "y_pctl_extroversion",          "do question group (normalized)",   "extroversion",         "y_intvl_extroversion"),
    ]

    for set in plot_settings:
        util.lin_reg_plot(dataset[set[0]], dataset[set[1]], set[2], set[3], dataset[set[4]])

def pca_plots(dataset):
    print("plots")
    
    dimensions = ["y_intvl_openness_2", "y_intvl_conscientiousness_2", "y_intvl_extroversion_2", "y_intvl_agreeableness_2", "y_intvl_neuroticism_2", "y_scie", "y_math", "y_sport", "y_ent", "y_hist", "y_geo", "y_arch"]
    subset = ['alfa', 'beta', 'scie', 'math', 'hist', 'arch', 'total_answers', 'total_categories']
    
    for dim in dimensions:
        print("running pca_2")
        run_pca_2(dataset, subset, dim)


def run_pca_2(dataset, subset, color_var):    
    # creating dataframe 
    df2 = dataset[configuration.pca_independents]
    
    print("head df 2")
    print(df2.head() )

    from sklearn.preprocessing import StandardScaler 
    
    scalar2 = StandardScaler() 
    
    # fitting 
    scalar2.fit(df2)
    scaled_data2 = scalar2.transform(df2)

    print("scaled data 2")
    print(scaled_data2.shape)
    print(scaled_data2)
    
    # Importing PCA 
    from sklearn.decomposition import PCA 

    pca2 = PCA() 
    pca2.fit(scaled_data2) 
    x_pca2 = pca2.transform(scaled_data2) 
    
    per_var = np.round(pca2.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    

    print(per_var)
    print(per_var[0] + per_var[1])

    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot for ' + color_var)
    plt.show()

    print(x_pca2.shape )


    # giving a larger plot 
    plt.figure(figsize =(8, 6)) 
    
    
    col = dataset[color_var]
    plt.scatter(x_pca2[:, 0], x_pca2[:, 1], c = col) 

    print(x_pca2[:, 0])
    
    # labeling x and y axes 
    plt.xlabel('First Principal Component') 
    plt.ylabel('Second Principal Component') 
    plt.title('PCA plot for' + color_var)

    plt.show()


    # alle mogelijke combinaties van configuration.dimensions loopen om te vinden welke combinatie de sterkste pc1 en pc2 geeft
    # om visueel beter te kunnen voorstellen 






def run_pca(dataset):    
    # creating dataframe 
    df2 = dataset[configuration.pca_independents]
    
    print("head df 2")
    print(df2.head() )

    from sklearn.preprocessing import StandardScaler 
    
    scalar2 = StandardScaler() 
    
    # fitting 
    scalar2.fit(df2)
    scaled_data2 = scalar2.transform(df2)

    print("scaled data 2")
    print(scaled_data2.shape)
    print(scaled_data2)
    
    # Importing PCA 
    from sklearn.decomposition import PCA 

    pca2 = PCA() 
    pca2.fit(scaled_data2) 
    x_pca2 = pca2.transform(scaled_data2) 
    
    per_var = np.round(pca2.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    

    print(per_var)
    print(per_var[0] + per_var[1])

    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    print(x_pca2.shape )


    # giving a larger plot 
    plt.figure(figsize =(8, 6)) 
    
    # y_intvl_openness	y_intvl_conscientiousness	y_intvl_extroversion	y_intvl_agreeableness	y_intvl_neuroticism	y_scie	y_math	y_sport	y_ent	y_hist	y_geo	y_arch
    col = dataset["y_intvl_openness"]
    plt.scatter(x_pca2[:, 0], x_pca2[:, 1], c = col) 

    print(x_pca2[:, 0])
    exit(0)
    
    # labeling x and y axes 
    plt.xlabel('First Principal Component') 
    plt.ylabel('Second Principal Component') 

    plt.show()


    # alle mogelijke combinaties van configuration.dimensions loopen om te vinden welke combinatie de sterkste pc1 en pc2 geeft
    # om visueel beter te kunnen voorstellen 





def calculate_best_pca_plots(dataset):
    for l in range(3, len(configuration.pca_independents)):
        highest = (0.0, [], [])
        for subset in itertools.combinations(configuration.pca_independents, l):
            new = calculate_pcn_sum(subset, dataset)
            if new[0] > highest[0]:
                highest = new
                #print("new highest:")
                #print(highest)
        print("level ", l, " highest = ", highest)

def calculate_pcn_sum(subset, dataset):
    df = dataset[sorted(subset)]
    from sklearn.preprocessing import StandardScaler 
    
    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)

    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)

    #print(subset)
    #print(per_var[0] + per_var[1])
    #print("################################################################################################################################################")

    return (per_var[0] + per_var[1], subset, per_var)





def run_pca_old(dataset):
    genes = ['gene' + str(i) for i in range(1,101)]
    
    wt = ['wt' + str(i) for i in range(1,6)]
    ko = ['ko' + str(i) for i in range(1,6)]
    
    data = pd.DataFrame(columns=[*wt, *ko], index=genes)

    print(data)
    
    for gene in data.index:
        data.loc[gene,'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
        data.loc[gene,'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    
    print(data)

    scaled_data = preprocessing.scale(data.T)
    print(data.shape)
    print(scaled_data.shape)
    
    pca = PCA() # create a PCA object
    pca.fit(scaled_data) # do the math
    pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data

    print(pca_data.shape)

    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    
    #the following code makes a fancy looking plot using PC1 and PC2
    pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)
    
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('My PCA Graph')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    
    plt.show()
 