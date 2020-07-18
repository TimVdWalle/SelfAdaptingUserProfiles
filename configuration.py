########################################################################
#   CONFIGURATION
########################################################################
# which parameters to calculate:
dependent   = [    'y_intvl_openness', 'y_intvl_conscientiousness', 'y_intvl_extroversion', 'y_intvl_agreeableness', 'y_intvl_neuroticism']

########################################################
#   filters
########################################################
age_min = 10
age_max = 99 

metric_low      = 15
metric_medium   = 25

########################################################
#   regression algorithm parameters
########################################################
corr_min        = 0.19
corr_sign_max   = 0.15

max_depth       = 11
max_dept_poly   = 20
max_dept_logreg = 10

kfold           = 10

pol_min_degree  = 1
pol_max_degree  = 6

pca_min_n  = 1
pca_max_n  = 45

map_threshold = 0.5

########################################################
#   random forest algorithm parameters
########################################################
rf_test_size                = 0.2
rf_loops                    = 40
rf_n_estimators             = 100
rf_max_depth                = 4         # max_depth = 2, 3 of 4


# for searching the optimal parameter values
rf_pt_n_estimators          = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256]
rf_pt_n_estimators_loop     = 4

import numpy as np
rf_pt_max_depths            = np.linspace(1, 20, 20, endpoint=True)
rf_pt_max_depth_loop        = 2


########################################################
#   content based parameters
########################################################
cb_threshold    = 2

########################################################
#   features & dimensions
########################################################
questions = ['wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4', 'spo1', 'spo2', 'spo3', 'spo4', 'spo6','ent1', 'ent2', 'ent3', 'ges1', 'ges2', 'ges3','aar1', 'aar2', 'aar3', 'aar4','arc1', 'arc2', 'arc3']


features = [ 
    ('alfa',    ['ges1', 'ges2', 'ges3', 'arc1', 'arc2', 'arc3']),
    ('beta',    ['wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4']),

    ('scie',    ['wet1', 'wet2', 'wet3', 'wet4', 'wet5']),
    ('math',    ['wis1', 'wis2', 'wis3', 'wis4']),
    ('sport',   ['spo1', 'spo2', 'spo3', 'spo4', 'spo6', 'arc2']),
    ('ent',     ['ent1', 'ent2', 'ent3']),
    ('hist',    ['ges1', 'ges2', 'ges3', 'arc2', 'arc3']),
    ('geo',     ['aar1', 'aar2', 'aar3', 'aar4', 'wet5']),
    ('arch',    ['arc1', 'arc2', 'arc3', 'ges1']),

    ('do_question',             ['wet3', 'wet4', 'spo2', 'spo3', 'aar1', 'aar2', 'arc1', 'aar3']),
    ('do_question_single',      ['wet3', 'spo2', 'aar1', 'arc1']),
    ('do_question_group',       ['wet4', 'spo3', 'aar2', 'aar3']),

    ('collect',                 ['wet5', 'spo6', 'ges2']),

    ('total_answers',           ['wet1','wet2','wet3','wet4','wet5','wis1','wis2','wis3','wis4','spo1','spo2','spo3','spo4','spo6', 'ent1','ent2', 'ent3','ges1','ges2','ges3','aar1','aar2','aar3','aar4','arc1','arc2','arc3']),
    ('total_categories',        ['scie', 'math', 'sport', 'ent', 'hist', 'geo', 'arch']),


    # grouped categories, manually determinded by studying their corrolations
    ('scie_sex',                ['sex', 'wet1', 'wet2', 'wet3', 'wet4', 'wet5']),
    ('collect_sex',             ['sex', 'wet5', 'spo6', 'ges2']),
    
    ('alfa_beta',               ['ges1', 'ges2', 'ges3', 'arc1', 'arc2', 'arc3', 'wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4']),
    ('alfa_science',            ['ges1', 'ges2', 'ges3', 'arc1', 'arc2', 'arc3', 'wet1', 'wet2', 'wet3', 'wet4', 'wet5']),
    ('alfa_math',               ['ges1', 'ges2', 'ges3', 'arc1', 'arc2', 'arc3', 'wis1', 'wis2', 'wis3', 'wis4']),
    ('alfa_sport',              ['ges1', 'ges2', 'ges3', 'arc1', 'arc2', 'arc3', 'spo1', 'spo2', 'spo3', 'spo4', 'spo6', 'wis4']),
    ('alfa_geo',                ['ges1', 'ges2', 'ges3', 'arc1', 'arc2', 'arc3', 'aar1', 'aar2', 'aar3', 'aar4', 'wet5']),
    ('alfa_doquestion',         ['ges1', 'ges2', 'ges3', 'arc1', 'arc2', 'arc3', 'wet3', 'wet4', 'spo2', 'spo3', 'aar1', 'aar2', 'aar3']),
    ('alfa_collect',            ['ges1', 'ges2', 'ges3', 'arc1', 'arc2', 'arc3', 'wet5', 'spo6']),
    
    ('beta_collect',            ['wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4',    'spo6', 'ges2']),
    ('beta_sport',              ['wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4', 'spo1', 'spo2', 'spo3', 'spo4', 'spo6', 'arc2']),
    ('beta_hist',               ['wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4', 'ges1', 'ges2', 'ges3', 'arc2', 'arc3']),
    ('beta_geo',                ['wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4', 'aar1', 'aar2', 'aar3', 'aar4']),
    ('beta_arch',               ['wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4', 'arc1', 'arc2', 'arc3', 'ges1']),
    ('beta_doquestion',         ['wet1', 'wet2', 'wet3', 'wet4', 'wet5', 'wis1', 'wis2', 'wis3', 'wis4', 'spo2', 'spo3', 'aar1', 'aar2', 'arc1', 'aar3'])
]

contentbased_interests = [
    ('scie',    ['wet1', 'wet2', 'wet3', 'wet4', 'wet5']),
    ('math',    ['wis1', 'wis2', 'wis3', 'wis4']),
    ('sport',   ['spo1', 'spo2', 'spo3', 'spo4', 'spo6', 'arc2']),
    ('ent',     ['ent1', 'ent2', 'ent3']),
    ('hist',    ['ges1', 'ges2', 'ges3', 'arc2', 'arc3']),
    ('geo',     ['aar1', 'aar2', 'aar3', 'aar4', 'wet5']),
    ('arch',    ['arc1', 'arc2', 'arc3', 'ges1'])
]

contentbased_psy = [
    ('extroversion',         ['spo2', 'spo3', 'aar3', 'wis3', 'wet4', 'aar4', 'aar3', 'arc1']),
    ('neuroticism',          ['wis4', 'arc2', 'aar3', 'wet3', 'ges1', 'aar3']),
    ('agreeableness',        ['spo1', 'spo2', 'aar3', 'wis3', 'ent1', 'ent2', 'spo4', 'spo6']),
    ('conscientiousness',    ['ges3', 'spo1', 'spo2', 'aar3', 'wis2', 'wis3', 'ent1', 'spo4', 'spo6', 'arc3', 'aar2', 'wet5']),
    ('openness',             ['ges3', 'wis4', 'spo4', 'ges1', 'arc3']),      #aar2, wet5
]



interestcategories      = [
                            ('y_scie',        'wetenschappen'),
                            ('y_math',        'wiskunde'),
                            ('y_sport',       'sport'),
                            ('y_ent',         'entertainment'),
                            ('y_hist',        'geschiedenis'),
                            ('y_geo',         'aardrijkskunde'),
                            ('y_arch',        'architectuur')
                        ]
    

#dimensions  = ['y_pctl_extroversion', 'y_pctl_neuroticism', 'y_pctl_agreeableness', 'y_pctl_conscientiousness', 'y_pctl_openness', 'y_intvl_openness', 'y_intvl_conscientiousness', 'y_intvl_extroversion', 'y_intvl_agreeableness', 'y_intvl_neuroticism', 'alfa_norm', 'beta_norm', 'scie_norm', 'math_norm', 'sport_norm', 'ent_norm', 'hist_norm', 'geo_norm', 'arch_norm', 'do_question_norm', 'do_question_single_norm', 'do_question_group_norm', 'collect_norm', 'total_answers_norm', 'total_categories_norm', 'scie_sex_norm','collect_sex_norm','alfa_math_norm','alfa_geo_norm','alfa_collect_norm','beta_hist_norm','age', 'sex', 'alfa', 'beta', 'scie', 'math', 'sport', 'ent', 'hist', 'geo', 'arch', 'do_question', 'do_question_single', 'do_question_group', 'collect', 'total_answers', 'total_categories', 'scie_sex','collect_sex','alfa_math','alfa_geo','alfa_collect','beta_hist','wet1', 'wet2', 'ges3', 'wis1', 'spo1', 'wis4', 'spo2', 'spo3', 'aar3', 'wis2', 'wis3', 'ent1', 'ent2', 'spo4', 'arc2', 'wet3', 'wet4', 'spo6', 'ges1', 'aar4', 'ent3', 'arc1', 'arc3', 'aar1', 'aar2', 'ges2', 'wet5']
#dimensions      = ['y_intvl_openness_2', 'y_intvl_conscientiousness_2', 'y_intvl_extroversion_2', 'y_intvl_agreeableness_2', 'y_intvl_neuroticism_2', 'y_scie','y_math','y_sport','y_ent','y_hist','y_geo','y_arch',    'y_pctl_extroversion', 'y_pctl_neuroticism', 'y_pctl_agreeableness', 'y_pctl_conscientiousness', 'y_pctl_openness', 'y_intvl_openness', 'y_intvl_conscientiousness', 'y_intvl_extroversion', 'y_intvl_agreeableness', 'y_intvl_neuroticism', 'alfa_norm', 'beta_norm', 'scie_norm', 'math_norm', 'sport_norm', 'ent_norm', 'hist_norm', 'geo_norm', 'arch_norm', 'do_question_norm', 'do_question_single_norm', 'do_question_group_norm', 'collect_norm', 'total_answers_norm', 'total_categories_norm','age', 'sex', 'alfa', 'beta', 'scie', 'math', 'sport', 'ent', 'hist', 'geo', 'arch', 'do_question', 'do_question_single', 'do_question_group', 'collect', 'total_answers', 'total_categories', 'wet1', 'wet2', 'ges3', 'wis1', 'spo1', 'wis4', 'spo2', 'spo3', 'aar3', 'wis2', 'wis3', 'ent1', 'ent2', 'spo4', 'arc2', 'wet3', 'wet4', 'spo6', 'ges1', 'aar4', 'ent3', 'arc1', 'arc3', 'aar1', 'aar2', 'ges2', 'wet5']
#dimensions      = [ 'y_pctl_extroversion', 'y_pctl_neuroticism', 'y_pctl_agreeableness', 'y_pctl_conscientiousness', 'y_pctl_openness',  'alfa_norm', 'beta_norm', 'scie_norm', 'math_norm', 'sport_norm', 'ent_norm', 'hist_norm', 'geo_norm', 'arch_norm', 'do_question_norm', 'do_question_single_norm', 'do_question_group_norm', 'collect_norm', 'total_answers_norm', 'total_categories_norm','age', 'sex', 'wet1', 'wet2', 'ges3', 'wis1', 'spo1', 'wis4', 'spo2', 'spo3', 'aar3', 'wis2', 'wis3', 'ent1', 'ent2', 'spo4', 'arc2', 'wet3', 'wet4', 'spo6', 'ges1', 'aar4', 'ent3', 'arc1', 'arc3', 'aar1', 'aar2', 'ges2', 'wet5']
dimensions  = ['y_intvl_openness', 'y_intvl_conscientiousness', 'y_intvl_extroversion', 'y_intvl_agreeableness', 'y_intvl_neuroticism', 'sex', 'age', 'alfa_norm', 'beta_norm', 'scie_norm', 'math_norm', 'sport_norm', 'ent_norm', 'hist_norm', 'geo_norm', 'arch_norm', 'do_question_norm', 'do_question_single_norm', 'do_question_group_norm', 'collect_norm', 'total_answers_norm', 'total_categories_norm', 'wet1', 'wet2', 'ges3', 'wis1', 'spo1', 'wis4', 'spo2', 'spo3', 'aar3', 'wis2', 'wis3', 'ent1', 'ent2', 'spo4', 'arc2', 'wet3', 'wet4', 'spo6', 'ges1', 'aar4', 'ent3', 'arc1', 'arc3', 'aar1', 'aar2', 'ges2', 'wet5']
dimensions_pol  = ['y_scie','y_math','y_sport','y_ent','y_hist','y_geo','y_arch',    'y_intvl_openness_2', 'y_intvl_conscientiousness_2', 'y_intvl_extroversion_2', 'y_intvl_agreeableness_2', 'y_intvl_neuroticism_2',   'y_pctl_extroversion', 'y_pctl_neuroticism', 'y_pctl_agreeableness', 'y_pctl_conscientiousness', 'y_pctl_openness', 'y_intvl_openness', 'y_intvl_conscientiousness', 'y_intvl_extroversion', 'y_intvl_agreeableness', 'y_intvl_neuroticism', 'sex', 'age', 'alfa_norm', 'beta_norm', 'scie_norm', 'math_norm', 'sport_norm', 'ent_norm', 'hist_norm', 'geo_norm', 'arch_norm', 'do_question_norm', 'do_question_single_norm', 'do_question_group_norm', 'collect_norm', 'total_answers_norm', 'total_categories_norm', 'wet1', 'wet2', 'ges3', 'wis1', 'spo1', 'wis4', 'spo2', 'spo3', 'aar3', 'wis2', 'wis3', 'ent1', 'ent2', 'spo4', 'arc2', 'wet3', 'wet4', 'spo6', 'ges1', 'aar4', 'ent3', 'arc1', 'arc3', 'aar1', 'aar2', 'ges2', 'wet5']

dimensions_logregpca = ['age', 'sex', 'alfa', 'beta', 'scie', 'math', 'sport', 'ent', 'hist', 'geo', 'arch', 'do_question', 'do_question_single', 'do_question_group', 'collect', 'total_answers', 'total_categories']



#dimensions_pol  = ['age', 'sex',   'alfa', 'beta', 'scie', 'math', 'sport', 'ent', 'hist', 'geo', 'arch', 'do_question', 'do_question_single', 'do_question_group', 'collect', 'total_answers', 'total_categories_norm', 'y_pctl_neuroticism', 'y_pctl_agreeableness', 'y_pctl_conscientiousness', 'y_pctl_openness', 'y_pctl_extroversion','wet1', 'wet2', 'ges3', 'wis1', 'spo1', 'wis4', 'spo2', 'spo3', 'aar3', 'wis2', 'wis3', 'ent1', 'ent2', 'spo4', 'arc2', 'wet3', 'wet4', 'spo6', 'ges1', 'aar4', 'ent3', 'arc1', 'arc3', 'aar1', 'aar2', 'ges2', 'wet5']
#dimensions  = ['age', 'sex', 'y_pctl_extroversion', 'alfa_norm', 'beta_norm', 'scie_norm', 'math_norm', 'sport_norm', 'ent_norm', 'hist_norm', 'geo_norm', 'arch_norm', 'do_question_norm', 'do_question_single_norm', 'do_question_group_norm', 'collect_norm', 'total_answers_norm', 'total_categories_norm', 'y_pctl_neuroticism', 'y_pctl_agreeableness', 'y_pctl_conscientiousness', 'y_pctl_openness', 'wet1', 'wet2', 'ges3', 'wis1', 'spo1', 'wis4', 'spo2', 'spo3', 'aar3', 'wis2', 'wis3', 'ent1', 'ent2', 'spo4', 'arc2', 'wet3', 'wet4', 'spo6', 'ges1', 'aar4', 'ent3', 'arc1', 'arc3', 'aar1', 'aar2', 'ges2', 'wet5']
all_independents_str = "alfa + beta + scie  + math  + sport  + ent  + hist  + geo  + arch  + do_question  + do_question_single  + do_question_group  + collect  + total_answers  + total_categories  + wet1  + wet2  + ges3  + wis1  + spo1  + wis4  + spo2  + spo3  + aar3  + wis2  + wis3  + ent1  + ent2  + spo4  + arc2  + wet3  + wet4  + spo6  + ges1  + aar4  + ent3  + arc1  + arc3  + aar1  + aar2  + ges2  + wet5"

#dependent   = ['y_pctl_extroversion', 'y_pctl_neuroticism', 'y_pctl_agreeableness', 'y_pctl_conscientiousness', 'y_pctl_openness']
#dependent   = ['y_scie','y_math','y_sport','y_ent','y_hist','y_geo','y_arch', 'y_pctl_extroversion', 'y_pctl_neuroticism', 'y_pctl_agreeableness', 'y_pctl_conscientiousness', 'y_pctl_openness',     'y_intvl_openness', 'y_intvl_conscientiousness', 'y_intvl_extroversion', 'y_intvl_agreeableness', 'y_intvl_neuroticism']

independent = ['sex',  'alfa_norm', 'beta_norm', 'scie_norm', 'math_norm', 'sport_norm', 'ent_norm', 'hist_norm', 'geo_norm', 'arch_norm', 'do_question_norm', 'do_question_single_norm', 'do_question_group_norm', 'collect_norm', 'total_answers_norm', 'total_categories_norm', 'wet1', 'wet2', 'ges3', 'wis1', 'spo1', 'wis4', 'spo2', 'spo3', 'aar3', 'wis2', 'wis3', 'ent1', 'ent2', 'spo4', 'arc2', 'wet3', 'wet4', 'spo6', 'ges1', 'aar4', 'ent3', 'arc1', 'arc3', 'aar1', 'aar2', 'ges2', 'wet5']
independent_major = ['sex', 'age', 'alfa_norm', 'beta_norm', 'scie_norm', 'math_norm', 'sport_norm', 'ent_norm', 'hist_norm', 'geo_norm', 'arch_norm', 'do_question_norm', 'do_question_single_norm', 'do_question_group_norm', 'collect_norm', 'total_answers_norm', 'total_categories_norm']

#pca_independents = ['age', 'alfa', 'beta', 'scie', 'math', 'sport', 'ent', 'hist', 'geo', 'arch', 'do_question', 'do_question_single', 'do_question_group', 'collect', 'total_answers', 'total_categories']
#pca_independents = ['age', 'sex', 'alfa', 'beta', 'scie', 'math', 'sport', 'ent', 'hist', 'geo', 'arch', 'do_question', 'do_question_single', 'do_question_group', 'collect', 'total_answers', 'total_categories']
pca_independents = ['alfa', 'beta', 'scie', 'math', 'hist', 'arch', 'total_answers', 'total_categories']





pschology_traits = [
    ('y_intvl_openness',                  'y_pctl_openness'),
    ('y_intvl_conscientiousness',         'y_pctl_conscientiousness'),
    ('y_intvl_extroversion',              'y_pctl_extroversion'),
    ('y_intvl_agreeableness',             'y_pctl_agreeableness'),
    ('y_intvl_neuroticism',               'y_pctl_neuroticism')
]


########################################################
#   datafile
########################################################
data_file = "enquete.csv"
data_file_cleaned = "enquete_cleaned.csv"
delete_rows = [60, 8, 7, 4, 3, 2, 1]
replace_empty = True
replace_val_false = False
replace_val_true = True

id_col = "profileId"
default_value = -999

idx_profile     = 0
idx_features    = 6
idx_traits      = 1

delete_cols = {
    'Tijdstempel',
    'Naam (niet verplicht)',
    'Woonplaats'
    #'Bekijk (in de app) een uitleg over hoe je reanimatie moet doen en bewaar deze bij je favorieten.'          # is double in the file because of changes in the form setup, is available as column in results, but contains no information
}

rename_cols = {
    ('age',                             'Leeftijd'), 
    ('sex',                             'Geslacht'), 

    ('y_pctl_extroversion',            'Welke score behaalde je voor Factor 1 : Extroversion '), 
    ('y_pctl_neuroticism',             'Welke score behaalde je voor Factor 2: Emotional stability'), 
    ('y_pctl_agreeableness',           'Welke score behaalde je voor Factor 3: Agreeableness'), 
    ('y_pctl_conscientiousness',       'Welke score behaalde je voor Factor 4: Conscientiousness'), 
    ('y_pctl_openness',                'Welke score behaalde je voor Factor 5: Intellect/Imagination'),

    ('wet1',                            'Rangschik de volgende elementen volgens opklimmend atoomgetal: '), 
    ('wet2',                            'Welk van de volgende elementen heeft de grootste atoommassa?'), 
    ('wet3',                            'Neem 5 koperen munten die dof of donker zijn.'), 
    ('wet4',                            'Doe de zwaartekracht-challenge. Daag je vrienden uit om zoveel mogelijk flessen op elkaar te stapelen zoals in dit filmpje (vanaf minuut 2) en post een foto van je resultaat. Degene met meeste flessen op elkaar is de winnaar.'), 
    ('wet5',                            'Bekijk (in de app) een filmpje over hoe een tsunami ontstaat en bewaar dit bij je favorieten.'), 

    ('wis1',                            'Bereken de volgende integraal:'), 
    ('wis2',                            'Men kan bewijzen dat P = NP '), 
    ('wis3',                            '2 is het kleinste even priemgetal'), 

    ('spo1',                            'Rangschik de judo-gordels volgens kleur'), 
    ('spo2',                            'Doe 30 sit-ups.'), 
    ('spo3',                            'Doe 30 sit-ups en post een selfie na je laatste sit-up.'), 
    ('spo4',                            'Welke stappen volg je bij reanimatie. Zet ze in de juiste volgorde.'), 
    #('spo5',                           'Bekijk (in de app) een uitleg over hoe je reanimatie moet doen en bewaar deze bij je favorieten.'), 
    ('spo6',                            'Bekijk (in de app) een uitleg over hoe je reanimatie moet doen en bewaar deze bij je favorieten..1'), 
    
    ('ent1',                            'Wie is de oudste van de Kardashian zussen'), 
    ('ent2',                            'Rangschik de volgende films volgens verschijningsdatum'), 
    ('ent3',                            'Hoeveel leden telt BTS ?'), 
    
    ('ges1',                            'Rangschik de Griekse zuilen chronologisch'),
    ('ges2',                            'Lees (in de app) waarom Adolf Hitler bijna Adolf Schicklgruber heette en bewaar dit tekstje bij je favorieten.'), 

    ('aar1',                            'Teken de kaart van België na en maak een foto van het resultaat.'), 
    ('aar2',                            'Een wedstrijd tegen je vriend. Schrijf beide zoveel mogelijke Amerikaanse staten op.'), 
    ('aar4',                            'Deze bergtop is de inspiratie achter het logo van de populaire chocoladereep Toblerone. In welk land staat deze berg?'), 

    ('arc1',                            'Fotografeer 3 verschillende bouwstijlen in je stad.'), 
    
    ('categories',                      'Ik ben geïnteresseerd in ')
}

rename_cols_indexed = {
    ('ges3',    12),            #Wie schreef de beroemde zin:  ‘Gallia est omnis divisa in partes tres
    ('wis4',    15),            #Hoeveel water kan er in een Olympisch zwembad
    ('aar3',    18),            #Maak een filmpje waar je strijdt tegen je vriend door om de beurt goeiedag te zeggen in een andere taal
    ('arc2',    24),            #Het Estadio Municipal de Braga is het voetbalstadion van de Portugese ploeg Braga
    ('arc3',    33)             #In welke stijl is deze lantaarnpaal, die typisch is voor Brussel
}

clean_values = {
    (replace_val_true,         'Deze vraag zou ik proberen op te lossen'),
    (replace_val_true,         'Deze vraag zou ik proberen uit te voeren'),
    (replace_val_true,         'Dit kaartje zou ik willen bewaren in mijn favorieten')    ,
    (1,         'Man')    ,    # only needed when using pearsonr instead of spearman
    (0,         'Vrouw')       # same
}