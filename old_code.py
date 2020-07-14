

#def lin_reg():
#    X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
#    Y = df['Stock_Index_Price']

    # Splitting the dataset into the Training set and Test set
    #from sklearn.model_selection import train_test_split 

3
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



    #print(dataset.iloc[:, 1:24])

    #R = np.corrcoef(dataset.iloc[:, 1:24], rowvar=True)
    #print(R)
    #pd.DataFrame(R).to_csv("corr.csv", index=False, encoding='utf8')

    #from termcolor import colored
    #print(colored('Hello, World!', 'white', 'on_red'))

    #R.to_csv("corr.csv", index=False, encoding='utf8')

    #for xaxis in xlist:
    #    for yaxis in ylist:
    #        #print(" ")
    #        lin_reg(xaxis[0], yaxis[0], xaxis[2], yaxis[1], xaxis[1])






# def lin_reg_old(x, y, xlabel, ylabel):
#     # linear regression fitting
#     lin_reg = LinearRegression()
#     lin_reg.fit(x, y)

#     # pearson cannot be used because data is not normally distributed
#     #print(xlabel + ':' + ylabel, stats.spearmanr(xsingle, y))

#     def viz_linear():
#         plt.scatter(x, y, color='red')
#         plt.plot(x, lin_reg.predict(x), color='blue')
#         plt.title(xlabel + ':' + ylabel)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.show()
#         return
#     viz_linear()
    


# def loop_all_combinations_for_old(dep, correlations, dataset):
#     # initialize vars with dummies and sentinels
#     highest = ('', 0, 0, 9999, 9999, '', [0], [0], 0)
#     highest_adj = ('', 0, 0, 9999, 9999, '', [0], [0], 0)
#     best_fit = ('', 0, 0, 9999, 9999, '', [0], [0], 0)

#     # first generate model with all independent variables
#     # generating model with all independents
#     new = regression_all(dep, configuration.independent, dataset)
#     highest = new
#     highest_adj = new
#     best_fit = new
#     display_result("model with all independents", new)

#     # second:  generate model with only the major independent variables
#     # generating model with all independents
#     new = regression_all(dep, configuration.independent_major, dataset)
#     display_result("model with only major independents", new)

#     print("creating all combinations for ", dep, "of :")
#     print(correlations[0])

#     # generate all combinations for the dimensions that are potentially correlation for dimension <dep>
#     for length in range(1, min(len(correlations[0]) + 1, configuration.max_depth +1)):
#         print("    generating length ", length)
#         for subset in itertools.combinations(correlations[0], length): 
#             independents = " + ".join(subset)         
#             #print("    ", dep, " => ", independents)
            
#             # new contains metrics as a tuple
#             # regression generates models and searches for best fit, highest rsquared, and rsquared_adjusted
#             new = regression_all(dep, subset, dataset)
#             if(new[1] > highest[1]):
#                 highest = new

#             if(new[2] > highest_adj[2]):
#                 highest_adj = new

#             if(new[4] < best_fit[4]):
#                 best_fit = new
    
#     display_result(dep + " highest squared", highest)
#     display_result(dep + " highest squared_adj", highest_adj)
#     display_result(dep + " best fit", best_fit)
   
#     #pd.DataFrame(best_fit[6]).to_csv(dep + "_y.csv", index=False, encoding='utf8')
#     #pd.DataFrame(best_fit[7]).to_csv(dep + "_ypred.csv", index=False, encoding='utf8')

# def regression_all(dep, subset, dataset):
#     #for dep in subset




#     for degree in range(configuration.pol_min_degree, configuration.pol_max_degree):
#         print("building models for degree", degree)
#         #regression(dep, subset, dataset, degree)

#     return ('', 0, 0, 9999, 9999, '', [0], [0], 0)





# def run_pca_old(dataset):  
#     ds = dataset[configuration.pca_independents]
#     #ds = ds.transpose()

#     print(ds)

#     #ds = data

#     scaled_data = preprocessing.scale(ds)

#     print(ds.shape)
#     print(scaled_data.shape)


#     pca = PCA() # create a PCA object
#     pca.fit(scaled_data) # do the math
#     pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data
#     print(pca_data.shape)

#     #The following code constructs the Scree plot
#     per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)

#     print("per_var")
#     print(per_var)

    
#     labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
#     plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
#     plt.ylabel('Percentage of Explained Variance')
#     plt.xlabel('Principal Component')
#     plt.title('Scree Plot')
#     plt.show()

#     print("pca_data")
#     print("____")
#     print(pca_data)
#     #the following code makes a fancy looking plot using PC1 and PC2
#     # pca_df = pd.DataFrame(pca_data, index=[configuration.pca_independents], columns=labels)
    
#     # plt.scatter(pca_df.PC1, pca_df.PC2)
#     # plt.title('My PCA Graph')
#     # plt.xlabel('PC1 - {0}%'.format(per_var[0]))
#     # plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    
#     # for sample in pca_df.index:
#     #     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    
#     # plt.show()

#     ## get the name of the top 10 measurements (genes) that contribute
#     ## most to pc1.
#     ## first, get the loading scores
#     loading_scores = pd.Series(pca.components_[0], index=configuration.pca_independents)
#     ## now sort the loading scores based on their magnitude
#     sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    
#     # get the names of the top 10 genes
#     top_10_genes = sorted_loading_scores[0:11].index.values
    
#     ## print the gene names and their scores (and +/- sign)
#     print(loading_scores[top_10_genes])




#     loading_scores = pd.Series(pca.components_[1], index=configuration.pca_independents)
#     ## now sort the loading scores based on their magnitude
#     sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    
#     # get the names of the top 10 genes
#     top_10_genes = sorted_loading_scores[0:11].index.values
    
#     ## print the gene names and their scores (and +/- sign)
#     print(loading_scores[top_10_genes])



#     loading_scores = pd.Series(pca.components_[2], index=configuration.pca_independents)
#     ## now sort the loading scores based on their magnitude
#     sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    
#     # get the names of the top 10 genes
#     top_10_genes = sorted_loading_scores[0:11].index.values
    
#     ## print the gene names and their scores (and +/- sign)
#     print(loading_scores[top_10_genes])