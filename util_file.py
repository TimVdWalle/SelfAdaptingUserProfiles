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
#   utilities (file) lib for SelfAdaptionUserProfiles
#
# ########################################################################

########################################################################
#   Imports
########################################################################
import pandas as pd
import configuration


########################################################################
#   Functions
########################################################################
def prepare_file(file):
    # read data from file
    print("reading datafile:", file)
    datafile = pd.read_csv(file)

    # clean file
    print("cleaning file")
    datafile = clean_file(datafile)

    # adding data, extracting features
    print("feature extraction")
    datafile = extract_features(datafile)

    # write cleaned up file
    print("writing cleaned file to", configuration.data_file_cleaned)
    datafile.to_csv(configuration.data_file_cleaned, index=False, encoding='utf8')


def clean_file(datafile_original):
    # make local copy
    datafile = datafile_original.copy()

    # remove rows with testdata
    for row in configuration.delete_rows:
        #print("dropping row", row)
        datafile.drop([datafile.index[row]], inplace = True)

    # filter on age
    datafile = datafile.loc[(datafile['Leeftijd'] >= configuration.age_min) & (datafile['Leeftijd'] <= configuration.age_max)]

    # renaming columns
    for ren_col in configuration.rename_cols:
        #print("renaming col ", ren_col[1], "to col ", ren_col[0])
        datafile.rename(columns={ren_col[1]: ren_col[0]}, inplace=True)

    for ren_col in configuration.rename_cols_indexed:
        #print("renaming col ", ren_col[1], "to col ", ren_col[0])
        datafile.rename(columns={ datafile.columns[ren_col[1]]: ren_col[0] }, inplace = True)

    # remove unnecessary columns
    for del_col in configuration.delete_cols:
        #print("removing col", del_col)
        datafile.drop(del_col, axis=1, inplace=True)

    # clean up values
    for rep_val in configuration.clean_values:
        #print("replacing val ", rep_val[1], "with ", rep_val[0])        
        datafile.replace(to_replace = rep_val[1], value = rep_val[0], inplace = True)

    # replace empty values
    if(configuration.replace_empty):
        #print("replacing empty values")
        datafile.fillna(configuration.replace_val_false, inplace=True)

    return datafile


def extract_features(datafile_original):
    # make local copy
    datafile = datafile_original.copy()

    # add column with profile identifier
    datafile.insert(0, configuration.id_col, 'profile')
    idx = configuration.idx_profile
    for i in datafile.index:
        idx = idx + 1
        datafile.at[i, configuration.id_col] = 'profile_' + str(idx)

    # update psychology traits with more useful values
    col_idx = configuration.idx_traits
    for new_col in configuration.pschology_traits:
        print("inserting col:", new_col[0])
        datafile.insert(col_idx, new_col[0], configuration.default_value)
        datafile.insert(col_idx, new_col[0] + "_2", configuration.default_value)
        col_idx = col_idx + 1

        # if in 33 percentile: -1
        # if between 33 and 66 percentile: 0
        # if more than 66 percentile: 1

        # update: changed: has 10 intervals now
        for i in datafile.index:
            #new_val = round(datafile.at[i, new_col[1]] / 50, 0) * 10
            #print(new_val)
            new_val = 0
            if(datafile.at[i, new_col[1]] > 66):
                new_val = 2
            elif (datafile.at[i, new_col[1]] >=50):
                new_val = 1
            
            datafile.at[i, new_col[0]] = new_val


            new_val = 0
            if(datafile.at[i, new_col[1]] > 50):
                new_val = 1
            elif (datafile.at[i, new_col[1]] <=50):
                new_val = 0
            
            datafile.at[i, new_col[0] + "_2"] = new_val

    # adding cols for interestcategories
    col_idx = configuration.idx_features
    for intcat in configuration.interestcategories:
        # adding dimension
        print("inserting col:", intcat[0])
        datafile.insert(col_idx, intcat[0], 0)
        col_idx = col_idx + 1

        for i in datafile.index:
            val = datafile.at[i, 'categories']
            if val != True and val != False and intcat[1] in val: 
                datafile.at[i, intcat[0]] = 1
                #print(datafile.at[i, intcat[0]], intcat[0])
            else:
                datafile.at[i, intcat[0]] = 0
                #print(datafile.at[i, intcat[0]], intcat[0])

    # extract features
    #col_idx = configuration.idx_features
    for feat in configuration.features:
        # adding dimension
        print("inserting col:", feat[0])
        datafile.insert(col_idx, feat[0], configuration.default_value)
        col_idx = col_idx + 1

        # adding normalized dimension
        print("inserting normalized col:", feat[0] + "_norm")
        datafile.insert(col_idx, feat[0] + "_norm", configuration.default_value * 1.0)
        col_idx = col_idx + 1
    


    for i in datafile.index:
        for feat in configuration.features:
            #print("checking values for ", feat[0])
            aggr = 0
            for answer in feat[1]:
                #print("checking answer", answer)
                if(datafile.at[i, answer] == True or datafile.at[i, answer] > 0):
                    #print("found value!")
                    aggr = aggr + 1
            #print(aggr)

            # adding value
            datafile.at[i, feat[0]] = aggr

            # adding normalized value
            #print(aggr, len(feat[1]), aggr / len(feat[1]))
            datafile.at[i, feat[0] + "_norm"] = aggr * 1.0 / len(feat[1])
            #print(datafile.at[i, feat[0] + "_norm"])
    return datafile
    
def namestr(obj):
    namespace = globals()
    return [name for name in namespace if namespace[name] is obj]