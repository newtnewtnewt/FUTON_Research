#!/usr/bin/env python
# coding: utf-8

# In[38]:


###
#  FUTON Model MDP + Q-Learning Creation Script
#  A Research Project conducted by Noah Dunn 
###

# Import the standard tools for working with Pandas dataframe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shelve
import random
import math
import time
import ctypes
import csv
# Pickle provides easy Object Serialization for quick read + writes of data
import pickle
# Vector Quantization for Determining Cluster Centers
from scipy.cluster.vq import vq
# Skikit offers a solution to perform K-Means++ clustering
from sklearn.cluster import KMeans
# Scipy provides a library to execute Z-Score Normalization
from scipy.stats import zscore
# We want to do type hinting for API clarification
from typing import *
# Itertools provides an easy way to perform Cartesian product on multiple sets
from itertools import product as cartesian_prod
import multiprocessing
# We need to perform 10 fold cross-validation
from sklearn.model_selection import StratifiedKFold
# For undersampling during the balancing phase
from imblearn.under_sampling import RandomUnderSampler
# For using PCA (Principal Component Analysis) to check variance
from sklearn.decomposition import PCA
### Some repetitive type hinting
int_matrix2D = np.array
float_matrix2D = np.array
int_matrix3D = np.array
float_matrix3D = np.array


# In[39]:


#  The Data File that will be used to conduct the experiments
patientdata:pd.DataFrame = pd.read_csv("/home/dunnnm2/thesis_research/rest_my_weary_head/patient_data_modified.csv")


# In[40]:


### 
#  An MDP, or Markov Decision Process is used to model relationships between various states and actions.
#  A state can be thought of in medical solution as a patient's diagnosis based on current vitals and state of being. 
#  An action can be thought of as a change in current diagnosis based on one of those vitals.
#  The inspirations for the bulk of this code came from Komorowksi's AI Clinician which can be found 
#  here: https://github.com/matthieukomorowski/AI_Clinician/blob/master/AIClinician_core_160219.m
###


# In[41]:


"""
# The extract_init_column_data function takes 6 arguments: 
# 
# patient_data:   The full DataFrame of all the patient data, raw and unfiltered
# id_column:      The name of the column in the dataframe containing the patient IDs
# binary_columns: A list containing the names of all the columns that are binary data (only 0s and 1s)
# normal_columns: A list containing the names of all the columns that are regular data (require no function transformations)
# log_columns:    A list containing the names of all the columns that are logarithmic data (Log has already been applied)
# debug_flag:     A boolean flag that indicates whether or not print statements will be executed
# 
# The function has 6 return values:
#
# colbin:       The resulting list of binary data columns, or the default list if none is provided
# colnorm:      The resulting list of normal data columns, or the default list if none is provided
# collog:       The resulting list of log data columns, or the default list if none is provided
# MIMIC_raw:    The DataFrame containing the data of all desired columns and their values
# id_count:     The total number of IDs to be used  
# icu_ids:      The ids of all patients to be used
# patient_idxs: 
"""

def extract_init_column_data(patient_data:pd.DataFrame, id_column:str='icustayid', binary_columns:List[str]=None, normal_columns:List[str]=None,
                            log_columns:List[str]=None, debug_flag:bool=False) -> (List[str], List[str], List[str], 
                                                                                   pd.DataFrame, pd.DataFrame, int, List[List[int]]):

    # Grab list of unique patient ICU stay IDs
    icu_ids:int = patient_data[id_column].unique()
    # Number of patients to be used for states
    id_count:int = icu_ids.size
    if debug_flag:
        print(id_count)

    # All our columns are broken up into 3 distinct categories:
    # 1. Binary values (0 or 1)
    # 2. Standard Ranges (Plain old Integers + Decimals)
    # 3. Logarthmic Values (columnvalue = log(columnvalue))
    colbin:List[str] = []
    colnorm:List[str] = [] 
    collog:List[str] = []
    
    # Enables custom column selection
    if binary_columns == None:
        colbin = ['gender','mechvent','max_dose_vaso','re_admission', 'qSOFAFlag', 'SOFAFlag']
    else:
        colbin = binary_columns
    
    if normal_columns == None:
        #colnorm = ['paO2', 'PaO2_FiO2', 'Platelets_count', 'GCS', 'MeanBP', 'SysBP', 'RR']
        colnorm = ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance', 'qSOFA'];
    else:
        colnorm = normal_columns
    if log_columns == None:
        # collog = ['Total_bili', 'Creatinine', 'output_total','output_4hourly']
        collog = ['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly'];
    else:
        collog = log_columns
    # Create seperate dataframes for each of the columns
    colbin_df:pd.DataFrame = patient_data[colbin]
    colnorm_df:pd.DataFrame = patient_data[colnorm]
    collog_df:pd.DataFrame = patient_data[collog]
    
    if debug_flag:
        # Let's make sure we have what we need
        print(colbin_df, "\n", colnorm_df, "\n", collog_df)
    # Rearrange the dataframe in order of binary, normal, and log data from left to right
    MIMIC_raw:pd.DataFrame = pd.concat([colbin_df, colnorm_df, collog_df], axis=1)
    if debug_flag:
        print(MIMIC_raw) 
    return colbin, colnorm, collog, MIMIC_raw, id_count, icu_ids
    


# In[42]:


"""
# The construct_zscores function takes 4 arguments: 
# 
# colbin:    The list of all columns representing binary values
# colnorm:   The list of all columns representing normal values
# collog:    The list of all columns representing log values
# MIMIC_raw: The DataFrame containing the data of all desired columns and their values
#
# and returns 
# colbin:    The resulting list of binary data columns, or the default list if none is provided
# colnorm:   The resulting list of normal data columns, or the default list if none is provided
# collog:    The resulting list of log data columns, or the default list if none is provided
# MIMIC_raw: The DataFrame containing the data of all desired columns and their values
"""

def construct_zscores(colbin:List[str], colnorm:List[str], collog:List[str], MIMIC_raw:pd.DataFrame, debug_flag:bool) -> pd.DataFrame:

    # We want a Z-Score for every item. This a measure of variance to see how far a value is from the mean

    # We need to normalize binaries to -0.5 and 0.5 for later use
    MIMIC_zscores:pd.DataFrame = MIMIC_raw

    # No need for the zscore algorithm here, -0.5 and 0.5 suffice
    MIMIC_zscores[colbin] = MIMIC_zscores[colbin] - 0.5

    # Recall these columns are logarithmic, so they needed converted back for proper Z-Scoring (+ 0.1 to avoid log(0))
    # Note that log(0.1) is essentially 0, Mathematically proved
    
    # zscore is the function pulled from the stats library in the initial import calls
    MIMIC_zscores[collog] = np.log(MIMIC_zscores[collog] + 0.1).apply(zscore)

    # Normal column requires no modifications. Z-Scores are calculated as normal
    MIMIC_zscores[colnorm] = MIMIC_zscores[colnorm].apply(zscore)
    if debug_flag:
        print(MIMIC_zscores)
    if 're_admission' in colbin:
        # We want the Re_Admission and fluid intake scaled Similarly to the other variables
        MIMIC_zscores['re_admission'] = np.log(MIMIC_zscores['re_admission'] + 0.6)
    if 'input_total' in collog:
        # Apply a scalar to fluid intake
        MIMIC_zscores['input_total'] = 2 * MIMIC_zscores['input_total']
    return MIMIC_zscores


# In[43]:


"""
In order to have a model severely skewed to over-predict death/life, it is desireable
to include an equal number of patients who have lived or died. 
Already with data that has been stratified based on averages, the balance gives the cleanest
possible situation to achieve balanced output

Input: 
train_set_predict: A DataFrame containing the IDs and summary data (mean, q1, q3, max, min)
for a given patient selected for the training set
train_set_response: A DataFrame containing information on whether or not a patient lived or died
is_debug: A flag that enables print statements in the functions

Output:
list_ids: The full list of IDs to be used for the training set

"""
def balance(train_set_predict:List[int], train_set_response:List[int], is_debug:bool) -> List[int]:
    # This will undersample the majority class using a specific seed to keep consistency across runs
    # In this case, that value is 47
    rus = RandomUnderSampler(random_state=47, sampling_strategy='majority')
    # train_set_predict = train_set_predict.reset_index()
    # Duplicate column names like are present here do not enable the sampling alg to work properly
    # Save the initial column headers to change back after the fact
    # Also, the reset_index shoves all the indexes to a 'level_0' column that will be used to identify them
    train_set_predict = train_set_predict.reset_index()
    start_column_headers = train_set_predict.columns
    train_set_modified = train_set_predict[:]
    # The fake column headers are just place holders so the randomized undersampling is allowed to take place
    train_set_modified.columns = ["fake" + str(i) for i in range(0, len(train_set_modified.columns))]
    # Perform the Undersampling of the majority class (Patients that Live)
    train_set_predict_bal, train_set_response_bal = rus.fit_resample(train_set_modified, train_set_response)
    # Reset column headers back to where they were 
    train_set_predict_bal.columns = start_column_headers
    # Validate that the number of columns is correct
    if is_debug:
        print(train_set_response_bal['death_state'].value_counts())
    list_ids = np.sort(train_set_predict_bal['level_0'].values.tolist())
    # Return the IDs that we intend to use for the training
    return list_ids


# In[44]:


"""
The golden standard for obtained ideal results for any machine learning exercise in the modern day is the
K-Fold Cross-Validation, specifically the stratified version of this technique. The K-fold divides the data into 10
seperate chunks chosen based on the feature set, and the stratified portion indicates that this data is relatively 
balanced in terms of the included data per fold. This function also calls the balance function for the training
output, in order to insure an equal number of patients who lived and patients who died are chosen

Input:
stratified_data: Time-Series DataFrame that has been summarized by patient in order to be used in the K-Fold
predictor_columns:  The columns to be used for stratification acting as predictors (I.E. the Summary Statistics for all Features)
response_column: The column that concerns performance, in this case the life/death status
is_debug: Enables print statements in the function

Output: 
all_train_sets: The List of Lists of all IDs to be used for the training data each run
all_test_sets: The List of lists of all IDs to be used for the testing data each run
"""

def crossval_split(stratified_data:pd.DataFrame, predictor_columns:List[str], 
                   response_column:List[str], is_debug:bool, id_correction_dict:Dict[int, int],
                   balance_flag:bool) -> (int_matrix2D, int_matrix2D):
    # The cross fold has an inner and outer component for proper division
    # This is provided by the sklearn library
    inner_cv = StratifiedKFold(10)
    outer_cv = StratifiedKFold(10)
    # Divy up the data into the new dataframes based on the selected columns
    predict_data = stratified_data[predictor_columns]
    response_data = stratified_data[response_column]
    # Save these to be appended to after all the cross folds are done
    all_train_sets = []
    all_test_sets = []
    # Iterate through using the cross-folds to build the correct IDs
    for training_samples, test_samples in outer_cv.split(predict_data, response_data):
        for inner_train, inner_test in inner_cv.split(predict_data.iloc[training_samples], response_data.iloc[training_samples]):
            # Grab the training data indexes and corresponding reponses
            train_predict = predict_data.iloc[training_samples].iloc[inner_train]
            train_response = response_data.iloc[training_samples].iloc[inner_train]
            # Grab the testing data indexes and corresponding responses
            test_predict = predict_data.iloc[training_samples].iloc[inner_test]
            train_ids = []
            if balance_flag:
                # Balance the training dataset, grab the desired ids
                train_ids = balance(train_predict, train_response, is_debug=False)
            else:
                train_ids = list(train_predict.index)
            # Grab the ids to use for testing
            test_ids = list(test_predict.index)
            # Fix the IDs to the correct indexes
            train_ids =list(map(id_correction_dict.get, train_ids))
            test_ids = list(map(id_correction_dict.get, test_ids))
            # Load them into a grand list
            all_train_sets.append(train_ids)
            all_test_sets.append(test_ids)
    return all_train_sets, all_test_sets


# In[45]:


"""
Normally in cross-fold validation, it is perfectly usable to insert a filtered dataset directly
into the cross-validation split function. Due to the nature of this data being time-series, with a variety of recorded 
chart events for each patient, a trick to still benefit from the cross-validation step is to 'flatten' the data
by providing summary statistics for each feature (mean, min, max, q1, q3), and form a stratification off that

Input:
patientdata: A DataFrame containing all the entire unflattened dataset
icu_ids: A list of all the ids of the patients

Output:
all_patient_summaries_df: A DataFrame with flattened values for each of the patients
"""
def flatten_timeseries_data(patientdata:pd.DataFrame, icu_ids:List[int], all_factors:List[str]) -> pd.DataFrame:
    # Iterate over all the factors for all the patients
    all_patient_summaries:List[pd.DataFrame] = []
    for id_val in icu_ids:
        # Grab each patient's full time-series run through the data
        single_patient_data = patientdata[patientdata['icustayid'] == id_val]
        # We can stratified time series data by 'flattening' it, constructing some summary statistic for each column
        desired_patient_columns = pd.DataFrame(single_patient_data, columns=all_factors)
        # Turn into dataframe and make each column seperate (Not 50 rows, 1 column)
        patient_mean_values = pd.DataFrame(desired_patient_columns.mean()).transpose()
        patient_min_values = pd.DataFrame(desired_patient_columns.min()).transpose()
        patient_max_values = pd.DataFrame(desired_patient_columns.max()).transpose()
        # Need to reset index on these due to funky interaction with concat
        patient_q1_values = pd.DataFrame(desired_patient_columns.quantile(0.25)).transpose().reset_index()
        patient_q3_values = pd.DataFrame(desired_patient_columns.quantile(0.75)).transpose().reset_index()
        # Build the dataframe from the pieces
        full_summary = pd.concat([patient_mean_values, patient_min_values, 
                                  patient_max_values, patient_q1_values, 
                                  patient_q3_values], axis=1, sort=False)
        # Add it to the master set
        all_patient_summaries.append(full_summary)
    # Fix the List into a dataframe
    # Also, set the indexes back to start at 1
    all_patient_summaries_df = pd.concat(all_patient_summaries, axis=0, sort=False).reset_index().drop(columns=['level_0'])
    return all_patient_summaries_df


# In[46]:


"""
As discussed in the cross_validate and balance functions, this step takes in the standard, preprocessed
dataframe of MIMIC data to be used for this experiment, and outputs Lists of balanced ID sets to be used.
The cross-validation and balancing steps produce a much better, and less biased model

Input:
icu_ids: DataFrame of IDs of all patients
debug_flag: Whether or not print statements are included
save_to_file: Whether or not intermediate steps are written to a file
regenerate_flag: When no files have been saved previously, set this to True to rerun all data
patientdata: The standard input MIMIC DataFrame

Output:
train_ids_set: A 2D list of all the lists of IDs to be used for training
test_ids_set:  A 2D list of all the lists of IDs to be used for testing
train_flag_set: A boolean representation of all the lists of IDs to be used for training
test_flag_set: A boolean representation of all the list of IDs to be used for testing

"""

def cross_validate_and_balance(icu_ids:pd.DataFrame, debug_flag:bool, save_to_file:bool, 
                                     regenerate_flag:bool, patientdata:pd.DataFrame, regenerate_cross:bool,
                                     all_factors:List[str]) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    if regenerate_flag:
        if not(regenerate_cross):
            # Flatten the full dataframe into summary statistics per patient
            # This is used to stratify and balance appropriately
            all_patient_summaries_df = flatten_timeseries_data(patientdata, icu_ids, all_factors)
            # This step messes up the indexes, we need to key the indexes to the correct ICU_IDS
            incorrect_indices = [i for i in all_patient_summaries_df.index]
            # Use dictionary comprehension to create a way of mapping the incorrect indices with 
            # their correct counterparts
            id_correction_dict = {incorrect_indices[i]: icu_ids[i] for i in range(0, len(incorrect_indices))}
            if debug_flag:
                print(all_patient_summaries_df)
            if save_to_file:
                    with open('pre_cross_data.txt', 'wb') as fp:
                        pickle.dump(all_patient_summaries_df, fp)
            # Grab the patient's 90d death statuses
            death_status = patientdata.drop_duplicates('icustayid')['mortality_90d']
            # To ensure everything is lined up adjacently, apply two more reset_index's
            # Also, the creation of the dataframe creates column indexes we don't desire. Get rid of these
            id_table = pd.DataFrame(icu_ids)
            id_table.columns = ['column_name']
            death_status = pd.DataFrame(death_status.values.tolist()).reset_index()
            death_status = death_status.drop(death_status.columns[0], axis=1)
            death_status.columns = ['death_state']
            icu_pair_set:pd.DataFrame = pd.concat([id_table, death_status, all_patient_summaries_df], axis=1, sort=False)
            if save_to_file:
                with open('interm_cross_data.txt', 'wb') as fp:
                        pickle.dump(icu_pair_set, fp)
                with open('id_correction_dictionary', 'wb') as fp:
                        pickle.dump(id_correction_dict, fp)
        # Remove duplicates from the columns list overall
        icu_pair_set = None
        id_correction_dict = None
        with open('interm_cross_data.txt', 'rb') as fp:
            icu_pair_set = pickle.load(fp)
        with open('id_correction_dictionary', 'rb') as fp:
            id_correction_dict = pickle.load(fp)
        predictor_columns = list(set(icu_pair_set.columns))
        # Death state is the response
        predictor_columns.remove('death_state')
        response_column = ['death_state']
        # Split the data using 10-fold Cross Validation and Balance the Training Dataset
        train_ids_set, test_ids_set = crossval_split(icu_pair_set, predictor_columns, response_column, 
                                                     is_debug=False, id_correction_dict=id_correction_dict, balance_flag=False)
        train_flag_set = []
        test_flag_set = []
        for train_ids in train_ids_set:
            train_flag_set.append(patientdata['icustayid'].isin(train_ids))
        for test_ids in test_ids_set:
            test_flag_set.append(patientdata['icustayid'].isin(test_ids))
        # Save the full data into files to be reused
        # There is no reason to redo the operations if nothing about the dataset has changed
        if save_to_file:
            with open('train_flags.txt', 'wb') as fp:
                pickle.dump(train_flag_set, fp)
            with open('test_flags.txt', 'wb') as fp:
                pickle.dump(test_flag_set, fp)
            with open('train_ids.txt', 'wb') as fp:
                pickle.dump(train_ids_set, fp)
            with open('test_ids.txt', 'wb') as fp:
                pickle.dump(test_ids_set, fp)
            
        return train_ids_set, test_ids_set, train_flag_set, test_flag_set
    else:
        # If the data has already been processed, load it from file
        train_ids_set = []
        test_ids_set = []
        train_flag_set = []
        test_flag_set = []
        with open('train_flags.txt', 'rb') as fp:
            train_flag_set = pickle.load(fp)
        with open('test_flags.txt', 'rb') as fp:
            test_flag_set = pickle.load(fp)
        with open('train_ids.txt', 'rb') as fp:
            train_ids_set = pickle.load(fp)
        with open('test_ids.txt', 'rb') as fp:
            test_ids_set = pickle.load(fp)
        return train_ids_set, test_ids_set, train_flag_set, test_flag_set


# In[47]:


"""
# calculate_all_hyperparameter_combos 
# 
# Input: N/A
# 
# Output: A 2D Float Matrix containing all the desired hyperparameters to test
# 
"""
def calculate_all_hyperparameter_combos() -> float_matrix2D:
    # Internal function to iterate by double
    def frange(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step
    # Gammas are incremented by 0.01
    gamma_values = [round(gamma, 2) for gamma in frange(0.01, 1.0, 0.01)]
    # Most common growth rates are 0.1, 0.05, and 0.01
    alpha_values = [0.1, 0.05, 0.01]
    # Return all possible combinations of hyperparameters using the cartesian product
    return list(cartesian_prod(gamma_values, alpha_values))


# In[48]:


"""
# The zscores_for_train_and_test function takes 8 arguments: 
# 
# train_flag:       The dataframe representing whether or not a row is used in training or not
# test_flag:        The dataframe representing whether or not a row is used in testing or not
# MIMIC_zscores:    The dataframe representing the zscores of the dataset
# debug_flag:       A flag that determines if print statements are executed
# save_flag:        A flag that determines if the training_zscores are saved
# patient_data:     The raw MIMIC dataframe
# bloc_name:        The name of the column that dictates a 'bloc' of time within an ICU visit
# id_name:          The name of the column that dictates an ICU stay's ID
# death_name:       The name of the column that dictates a patient's life status 
# 
# and returns 7 values:
# 
# train_zscores:   The portion of MIMIC_zscores that is in the training set
# test_zscores:    The portion of MIMIC_zscores that is in the testing set
# train_blocs:     The rows of the MIMIC dataset that is in the training set
# test_blocs:      The rows of the MIMIC dataset that is in the testing set
# train_id_list:   The list of all IDs in the training set
# train_90d:       A flag indicating if a given patient died in the training set after 90d
# test_90d:        A flag indicating if a given patient died in the testing set after 90d
"""

def zscores_for_train_and_test(train_flag:pd.DataFrame, test_flag:pd.DataFrame, MIMIC_zscores:pd.DataFrame, 
                               debug_flag:bool, save_flag:bool, patient_data:pd.DataFrame, bloc_name:str, 
                               id_name:str, death_name:str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                                pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                                pd.DataFrame, pd.DataFrame):

    # Seperate the Z-Scores for the training set and the testing set
    train_zscores:pd.DataFrame = MIMIC_zscores[train_flag]
    test_zscores:pd.DataFrame = MIMIC_zscores[test_flag]

    # The blocs of relevance in order based on the train and test set
    # These will be used to build relevant data frames later down
    train_blocs:List[int] = patientdata[train_flag][bloc_name]
    test_blocs:List[int] = patientdata[test_flag][bloc_name]

    # Doing the same with the patient ids
    train_id_list:pd.DataFrame = patientdata[train_flag][id_name].reset_index()['icustayid']
    test_id_list:pd.DataFrame = patientdata[test_flag][id_name].reset_index()['icustayid']
    # We modify a column later to use for the test_id_list, not necesssary here.

    # Grabbing the boolean values for the patients who died within 90 days in the training set
    train_90d:pd.DataFrame = patientdata[train_flag][death_name]
    test_90d:pd.DataFrame = patientdata[test_flag][death_name]
    

    if save_flag:
        # Python has object serialization to make write/reads fasters, in the form of pickle
        # Save the important data (clusters created as a result of the K-Means operations)
        # This process takes quite a while. This will provide a checkpoint to decrease compute time
        # until the code is put into dev.
        with open('train_zscores.txt', 'wb') as fp:
            pickle.dump(train_zscores, fp)
        with open('test_zscores.txt', 'wb') as fp:
            pickle.dump(test_zscores, fp)
    return train_zscores, test_zscores, train_blocs, test_blocs, train_id_list, test_id_list, train_90d, test_90d


# In[49]:


"""
# determine_optimal_clusters is a method that uses the modern curvature method to optimize clusters in the KMeans
# clustering algorithm. The data for this is provided by euclidean variance calculated using Redhawk supercluster
#
# The input is:
# variance_data: A list containing the total variance observed at each k-value (number of clusters)
# show_graph: whether or not we would like to display the graph of the curvature
# 
# The output is:
# optimal_cluster_count: The int representing the optimal number of clusters
"""
def determine_optimal_clusters(variance_data:List[float], show_graph:bool): 
    # A large portion of this was provided/inspired by code from Dr. Giabbanelli's machine learning course
    lower_bound:int = 1
    upper_bound:int = len(variance_data) + 1
    val_range = range(lower_bound, upper_bound)
    # Approximating the values to a polynomial fit
    coefs:List[float] = np.polyfit(val_range, variance_data, 3)
    # Generate a list of values at each point
    coefs_vals:List[float] = np.polyval(coefs[::1], val_range)
    if show_graph:
        # Generate the plot
        plt.plot(val_range, variance_data)
        plt.show()
    # Curve values fluctuate based on geometric scaling, as such, it is required
    # to test different k-values across different alphas
    alphas:List[float] = [i/20.0 for i in range(0, 400)]
    max_curve:float = -1
    max_k:float = -1
    # Test on a variety of Alphas and find the maximal result
    for alpha in alphas:
        # Scale the variation values, equation coefficients, and curve values based on various alphas
        scaled_vars:List[float] = [alpha * variance_val for variance_val in variance_data]
        scaled_coefs:List[float] = np.polyfit(val_range, scaled_vars, 3)
        # Calculate the curvature of the line at each step
        curve_vals = np.polyder(scaled_coefs)
        curve_vals_mod = np.polyder(curve_vals)
        scaled_curves:List[float] = []
        # The authors of the curvature method use these two tranformations to calculate values
        for k in val_range:
            function_val_one:float = abs(np.polyval(curve_vals_mod, k))
            function_val_two:float = 1 + np.polyval(curve_vals, k)**2
            scaled_curves.append(function_val_one / (function_val_two**1.5))
        # Iterate over all the scaled curves to determine the max
        index_and_value = max(enumerate(scaled_curves), key=(lambda x: x[1]))
        max_index:float = index_and_value[0]
        max_value:float = index_and_value[1]
        if(max_value > max_curve):
            max_curve = max_value 
            max_k = max_index
    # The k value needs to be scaled back to it's actual value, not its list index
    true_k = max_k + 1
    return true_k
        


# In[50]:


def calculate_optimal_clusters(train_zscores:pd.DataFrame, max_state_count:int=750, num_loops_per_iter:int=10000,
                                max_num_iter:int=32):
    # Code for this sample was provided largely by Dr. Giabbanelli 
    # This makes use of the new curvature method for calculating optimal clusters
    # For K-Means sampling
    variance_results:List[float] = np.zeros((max_state_count))
    for state_count in range(1, max_state_count):
        clusters_models = KMeans(n_clusters=state_count, max_iter=num_loops_per_iter, n_init=max_num_iter).fit(train_zscores)
        cluster_values = clusters_models.cluster_centers_
        closest_clusters:np.ndarray = vq(train_zscores, cluster_values)
        cluster_distances = closest_clusters[1]
        total_variance = 0
        for i in range(0, len(cluster_distances)):
            total_variance = total_variance + cluster_distances[i]
        variance_results[state_count] = total_variance    
        print(f'Finished with {state_count} at a variance of {total_variance}')
    with open('variance_full_results.txt', 'wb') as fp:
            pickle.dump(variance_results, fp)
    return variance_results


# In[51]:


def calculate_optimal_clusters_parallel(num:int):
    data_set=sample_train_set
    max_state_count=state_count
    num_loops_per_iter=10000
    max_num_iter=32
    total_needed_runs:List[int] = [i for i in range(1, 751)]
    thread_needed_runs:List[int] = np.array_split(total_needed_runs, 24)[num]
    # Code for this sample was provided largely by Dr. Giabbanelli 
    # This makes use of the new curvature method for calculating optimal clusters
    # For K-Means sampling
    variance_results:List[float] = np.zeros((max_state_count))
    for state_count in thread_needed_runs:
        clusters_models = KMeans(n_clusters=state_count, max_iter=num_loops_per_iter, n_init=max_num_iter).fit(sample_train_set)
        cluster_values = clusters_models.cluster_centers_
        closest_clusters:np.ndarray = vq(train_zscores, cluster_values)
        cluster_distances = closest_clusters[1]
        total_variance = 0
        for i in range(0, len(cluster_distances)):
            total_variance = total_variance + cluster_distances[i]
        variance_results[state_count] = total_variance    
        print(f'Finished with {state_count} at a variance of {total_variance}')
        single_run = f'{state_count},{total_variance}'
        with open('all_cluster_runs.csv', 'a') as f:
            print(single_run, file=f)
            


# In[52]:


"""
# calculate_optimal_clusters_driver runs either a single or multithreaded variant of calculate_optimal_clusters,
# determining the optimal_cluster count up to max_state_count
#
# Input:
# data_set - zscores for the training data
# run_optimal_clusters - Whether to run or pull directly from already saved data
# run_multithread - Whether or not to multithread the process (This does not work on Jupyter)
# show_graph - Whether or not to print a graph of the action distribution
# thread_count - How many threads to use in the case of multithreading
# max_state_count - The maximum number of threads to calculate up to
# 
# Output:
# optimal_cluster_count - The optimal number of clusters calculated according to the Curvature method
"""

def calculate_optimal_clusters_driver(data_set:pd.DataFrame, run_optimal_clusters:bool, run_multithread:bool, 
                                      show_graph:bool, thread_count:int, max_state_count:int) -> int:
    # Cluster selection
    optimal_cluster_count = 0 
    # If we wish to find the optimal number of clusters
    if run_optimal_clusters:
        # This code is here for posterity, and cannot be run natively in juptyer. Use the stock Python CLI command or 
        # PyPi to actually run this. Note, it can take forever with large Nodes
        if run_multithread:     
            pool = multiprocessing.Pool()
            pool.map(calculate_optimal_clusters_parallel, range(23))
        else:
            cluster_results = calculate_optimal_clusters(train_zscores=data_set, max_state_count=max_state_count, num_loops_per_iter=10000, 
                                max_num_iter=32)
            optimal_cluster_count = determine_optimal_clusters(cluster_results, show_graph=show_graph)
            return optimal_cluster_count
    # Load the data from the cluster variances into here
    cluster_results = []
    with open('all_cluster_runs.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            cluster_results.append(float(row[1]))
    optimal_cluster_count = determine_optimal_clusters(cluster_results, show_graph=show_graph)
    return optimal_cluster_count
    


# In[53]:


"""
# The kmeans_cluster_calculations function takes 5 arguments: 
# data_set:               The training dataset used to generate the various clusters
# state_count:            Total number of clusters desired in the end result
# num_loops_per_iter:     The number of loops ran per 1 iteration to approximate clusters
# max_num_iter:           Number of iterations to run the algorithm
# debug_flag:             True indicates print statements are included, False indicates no prints
# regenerate_flag:        True indictates the kmeans will be run again. False means the cached/previous file value will be used
# 
# and returns 4 values:
# 
# cluster_values:         A value generated based on the Z-Scores of all the patient's data relative to each other
# cluster_labels:         A number generated for the correspond state -> value relationship. Range is 0 to state_count
# train_zscores:          The Z-Scores of the training set as created earlier
# closest_clusters:       A list representing which state each patient datapoint is closest to
# 
"""
def kmeans_cluster_calculations(train_zscores:pd.DataFrame, test_zscores:pd.DataFrame,
                                max_num_iter:int=32, state_count:int=750, num_loops_per_iter:int=10000,
                                debug_flag:bool=True, regenerate_flag=True, load_zscores=False) -> (List[List[np.float64]], List[int], pd.DataFrame, 
                                                                                                     np.ndarray, np.ndarray):
    # In order to prepare a proper set of states, we want to use k-means clustering to group various patients into 
    # distinct states based on Z-Scores

    # K-Means or K-Means++ is a technique used to condense very diverse and sparse data into similar groups called 'clusters'
    # The K-means algorithm will create k clusters from N data points. In the case of this research,
    # the algorithm divides patients into groups that have similar data (age, blood pressure, etc..) and creates a faux 'point'
    # at the center of that particular clustering of data

    # The KMeans takes three 'settings' arguments
    # 1. n_clusters: The number of clusters (later to be used as states), that we desire the algorithm to produce
    # this value has been preset to state_count which is 750
    # 2. max_iter: How many times each round of k-means clustering will make adjustments, set at 10,000 in my case
    # 3. n_init: The number of max_iter batches that will be conducted in a row. The best of these will be chosen
    # and saved in the variable clusters_models
    cluster_models = []
    cluster_labels:List[int] = [] 
    cluster_values:List[List[np.float64]] = []
    if regenerate_flag:
        clusters_models = KMeans(n_clusters=state_count, max_iter=num_loops_per_iter, n_init=max_num_iter).fit(train_zscores)
        # Save the important data (clusters created as a result of the K-Means operations)
        # This process takes quite a while. This will provide a checkpoint to decrease compute time
        # until the code is put into dev.
        cluster_labels = clusters_models.labels_
        cluster_values = clusters_models.cluster_centers_
        with open('cluster_labels.txt', 'wb') as fp:
            pickle.dump(clusters_models.labels_, fp)
        with open('cluster_centers.txt', 'wb') as fp:
            pickle.dump(clusters_models.cluster_centers_, fp)
        if debug_flag:
            print(clusters_models.labels_)
            print(clusters_models.cluster_centers_)
    

    

    with open ('cluster_centers.txt', 'rb') as fp:
        cluster_values = pickle.load(fp)
    with open ('cluster_labels.txt', 'rb') as fp:
        cluster_labels = pickle.load(fp)
    if load_zscores:
        with open ('train_zscores.txt', 'rb') as fp:
            train_zscores = pickle.load(fp)
        with open ('test_zscores.txt', 'rb') as fp:
            test_zscores = pickle.load(fp)
    
    if debug_flag:
        print(cluster_values, "\n", "Dimensions: ", len(cluster_values)," x ", len(cluster_values[0]), "\n", train_zscores)
    
    # We now want to use the clusters to determine their nearest real data point neighbors
    # As a visual of this. Suppose we have 4 flags of different colors scattered over a park. The K-Means++ algorithm
    # is what planted the flags in the middle of groups of people that are similar. The KNN Search (K nearest neighbor search)
    # can be used in MatLab as a simple point finder instead of as a more complicated Supervised Learning algorithm. In Python 
    # we can make use of the Vector Quanization (vq) package to assign each point to a centroid
    closest_clusters:np.ndarray = vq(train_zscores, cluster_values)
    closest_clusters_test:np.ndarray = vq(test_zscores, cluster_values)
    

    if debug_flag:
        print(len(closest_clusters[0]))

    # As an aside, closest_clusters[1] contains the distance between each point's values (in this case 50 of them)
    # and their closest cluster's values.
    # Ex: If a point is [1, 1, 1] and it's closest cluster is the point [3, 3, 3]  closest_clusters[1] would contain the vector
    # [abs(3 - 1), abs(3 - 1), abs(3 - 1)] or [2, 2, 2]

    # Validate that all the points are in the range 0-749 (since there are only 750 clusters as specified previously)
    for i in closest_clusters[0]:
        if(i > (state_count - 1) or i < 0):
            print("The clusters you are searching for are not configured properly and are out of bounds")
            print("Did you modify the cluster_count variable without changing this error configuration?")
    
    return cluster_values, cluster_labels, train_zscores, closest_clusters, closest_clusters_test


# In[54]:


"""
# The generate_action_column function takes 4 arguments: 
#
# column_values: A series of column values from a dataframe that we want to turn into action states
# num_groups: How many groups or distinct actions we want to split the data into
# column_name: The name of the column used for print debug statements
# num_rows: The total number of rows in the full column before modifications (This is normally patientdata[column_name].size)
# 
# This function returns column_actions, a series that represents the 'action', or group that each row of data falls under.
#
# An example is found down below, but in words, this function takes a full column of data, groups 
# the values for that data into num_groups distinct actions, and returns a series representing actions based on row
# 
# Ex: Patients' blood pressure might be grouped into 5 categories (Action 1: < 20 mmHg, Action 2: > 20 mmHg && < 60 mmHg... etc)
"""

def generate_action_column(column_values:pd.DataFrame, num_groups:int, column_name:str, num_rows:int, debug_flag:bool=True) -> pd.Series:
    # Determine minimum and maxium to scale data appropriately
    if debug_flag:
        print("Old Lowest ", column_name, " Rank: ", min(column_values.rank()))
        print("Old Highest " , column_name,  " Rank: ", max(column_values.rank()))
    # Now we want to rank these actions in order of their value (lowest to highest)
    # Normalizing according to lowest and highest rank
    
    # Moving the minimum to zero
    column_ranks:pd.Series = (column_values.rank() - min(column_values.rank()))
    # Shifting the max to approximately 1.0
    column_ranks:pd.Series = column_ranks / max(column_ranks)
    
    if debug_flag:
        # Validate that the range is indeed 0 to 1
        print("New Lowest ", column_name, " Rank: ", min(column_ranks))
        print("New Highest ", column_name, " Rank: ", max(column_ranks))

    # The Max of all column values needs to be nearly 1, and the min of all column
    # values needs to be nearly 0 
    if round(max(column_ranks), 3) != 1 or round(min(column_ranks), 3) != 0:
        print("The ranks are not normalized correctly, either the max is too high, or the minium is too low")
        print("Current max: ", round(max(column_ranks), 3))
        print("Curret min: ", round(min(column_ranks), 3))
    # Normalize the rank values to even intevals of ranks
    old_values:List[float] = np.sort(column_ranks.unique()).tolist()
    even_intervals:List[float] = [i/column_ranks.unique().size for i in range(0, column_ranks.unique().size)]
    # Iterate over the Series to apply the normalized values
    for i in range(0, column_ranks.size):
        old_value_index:float = old_values.index(column_ranks[i])
        column_ranks[i] = even_intervals[old_value_index] 
    
    # This is a mathematics trick to seperate all the values into {num_groups} distinct groups based on their rank.
    # Given different columns of interest this can take different forms. For IV fluids, this number is 5.
    column_groups:pd.Series = np.floor(((column_ranks + 1.0/float(num_groups)) * (num_groups - 1))) + 1
    
    # Validate that groups are all associated with desired group split
    if not(column_groups.isin([i for i in range(1, num_groups + 1)]).any()):
        print("Groups chosen fall outside the desired 1-" + num_groups + " window")

    column_actions:pd.Series = pd.Series([1 for i in range(0, num_rows)])

    # If the value was non-zero and grouped in the 1 - 4 groups, we grab its value to save as an action
    for index in column_groups.index:
        column_actions[index] = column_groups[index]
        
    return column_actions
    


# In[55]:


"""
# This function takes two arguments:
# actions_column: A column of action groups generated by the above function (generate_action_column())
# real_values: The actual values from the dataset corresponding to the same column as actions_column
# and returns a list that contains the real median values for each 'group' actions.
#
# Ex: We apply the function to the action_column "IV_Fluid", which has split the data into 4 different groups of 
# IV_Fluid actions. This function will produce a list containing the median amount of IV_Fluid administered for each of those
# groups (Group 1 -> Adminster 20 mL, Group 2 -> Administer 40 mL, Group 3 -> Administer 60 mL, Group 4 -> Administer 80 mL
"""

def median_action_values(actions_column: pd.DataFrame, real_values:pd.DataFrame) -> List[np.float64]:
    # Grab all the unique actions for a column and sort them
    all_groups:List[np.float64] = np.sort(actions_column.unique())
    # Concatanate the group number and real value for each row
    action_set:pd.DataFrame = pd.concat([actions_column, real_values], axis=1, sort=False)
    # Name the columns for accurate querying
    action_set.columns = ['group_id', 'data_val']
    # Grab the median value for each group based on group number using python list comprehension
    median_values:List[np.float64] = [np.median(action_set[action_set['group_id'] == i]['data_val']) for i in all_groups]
    return median_values


# In[56]:


""" 
# This function takes one argument
# list_action_columns: This is a Pandas dataframe that contains all the action_columns we desir to be grouped by index
# This can be retrieving using the previously defined 'median action' function 
# 
# and returns two items:
# list_action columns: The 'keys' or integers that represent every permutation of actions
# chosen_action: The key that was chosen based on the action values in each column
"""
def generate_action_matrix(list_action_columns:pd.DataFrame) -> (List[int], List[int]):
    # Grabs the list of columns the user has provided for use
    desired_columns:List[str] = [column for column in list_action_columns]
    # Drops all group combinations that are duplicates
    list_action_columns_indexes:pd.DataFrame = list_action_columns.drop_duplicates(desired_columns)
    # Sorts all combinations in order
    list_action_columns_indexes = list_action_columns_indexes.sort_values(desired_columns)
    # Create a dictionary based on the values from the dataframe 
    list_action_columns_indexes:List[int] = list_action_columns_indexes.values.tolist() 
    # Determine which index in the list each row corresponds to 
    # Ex: For an 2-D action permutation list of [1,1] thru [5,5], there are 5 x 5 possibilities
    # {1..5}, {1..5}, so there are 25 possible permutations, the indexes will run 1 - 25
    chosen_action:List[int] = [list_action_columns_indexes.index(val_pair) for val_pair in list_action_columns.values.tolist()]
    # Return the keys first, and then the true values for the dataset
    return list_action_columns_indexes, chosen_action
    


# In[57]:


"""
# The build_dataset_actions function takes 7 arguments: 
# patientdata: A Dataframe representing all the patients and their data
# first_column: Our first column that we desire to build our action matrix with (first_column x second_column) (SOFA by default)
# second_column: Our second column that we desire to build or action matrix with (first_column x second_column) (qSOFA by default)
# num_groups_first_column: The number of groups we want to divide our first column into (5 by default)
# num_groups_second_column: The number of groups we want to divide our second column into (4 by default)
# debug_flag: A boolean that determines whether we want debug prints presents
# graph_flag: Whether we want to print the graph or not
# train_flag: The List representing which rows have been marked for training
# 
# and returns 3 values:
# 
# train_chosen_actions:        The actions for each row of the training dataset
# train_action_values:         The matrix representing (first_column x second_column)
# action_list:                 A list of actions taken at every given datapoint
"""
def build_dataset_actions(patientdata:pd.DataFrame, first_column:str="SOFA", second_column:str="qSOFA", num_groups_first_column:int=5,
                         num_groups_second_column:int=4, debug_flag:bool=True, graph_flag:bool=True,
                         train_flag:pd.Series=[], test_flag:pd.Series=[]) -> (pd.Series, List[List[int]], List[int]):
    # Generate the actions for a the first desired column of data
    first_col:pd.DataFrame = patientdata[first_column]
    first_col_actions:pd.Series = generate_action_column(column_values = first_col, num_groups = num_groups_first_column, 
                                                        column_name = first_column, 
                                                         num_rows = patientdata[first_column].size, debug_flag=debug_flag)
    if debug_flag:
        print(first_col_actions.unique())    
    # Now do the same for the second desired column of data
    second_col:pd.DataFrame = patientdata[second_column]
    second_col_actions:pd.Series = generate_action_column(column_values = second_col, num_groups = num_groups_second_column,  
                                                         column_name = second_column, 
                                                         num_rows = patientdata[second_column].size, debug_flag=debug_flag)
    if debug_flag:
        print(second_col_actions.unique())
    
    # Obtain the median real values that each division represents
    first_median_actions:List[np.float64] = median_action_values(actions_column = first_col_actions, real_values = patientdata[first_column])
    second_median_actions:List[np.float64] = median_action_values(actions_column = second_col_actions, real_values = patientdata[second_column])
    
    if debug_flag:
        print(first_column," Action Median Values:", str(first_col_actions), "\n" + second_column + ":", second_median_actions, "\n")
    ###
    # FINISH CONSTRUCTION OF ALL ACTIONS AND THEIR VALUES
    ###
    # Combine the columns that we desire to observe (iv_fluid_actions, vasopressor_actions)
    combined_groups:pd.DataFrame = pd.concat([first_col_actions, second_col_actions], axis=1, sort=False)
    # Name the columns for proper usage in the function
    combined_groups.columns = [first_column, second_column]
    
    # The Key value pair for every datapoint and the corresponding action taken at that point
    action_keys, action_list = generate_action_matrix(list_action_columns = combined_groups)
    # Plot the distribution of actions
    if graph_flag:
        plt.hist(action_list, density=False, bins=20)  # `density=False` would make counts
        plt.ylabel("Count")
        plt.xlabel("Index of Action Chosen: 1 through 24")
    # Grab a Series representing the action taken by the train data only
    train_chosen_actions:pd.Series = pd.Series(action_list)[train_flag]
    # Grab a Series representing the action taken by the test data only
    test_chosen_actions:pd.Series = pd.Series(action_list)[test_flag]

    # Assign all action choices to their corresponding median values as shown previously
    if debug_flag:
        print(first_median_actions, second_median_actions)


    # This gives us the representative median values for a patient's vitals present in various action groups
    # action_keys[i] corresponds to train_action_values[i]
    # So, if the patient falls into group [1, 1] or no iv fluid given, no vasopressor administered,
    # The corresponding median values for this group will be represented by train_action_values (0.0, 0.0).
    # A patient in group [1, 2] (no iv fluid, a little vasopressor) will have a median real value of (0.0, 0.04)
    action_values_matrix:List[List[int]] = list(cartesian_prod(first_median_actions, second_median_actions))

    if len(action_values_matrix) != len(first_median_actions) * len(second_median_actions):
        print("Something went wrong in determining the Cartesian product")
    
    action_count:int = len(action_values_matrix)
    return train_chosen_actions, test_chosen_actions, action_values_matrix, action_list, action_count
    


# In[58]:


"""
# The construct_prestate_matrix_train function takes 5 arguments: 
# train_90d:              A column representing whether or not a patient was dead or alive at the end of 90days
# train_blocs:            All the rows of data for a patient for each individual hospital stay
# closest_clusters:       The closest data cluster that a given data point falls near
# train_chosen_actions:   The action that is represented by two of the patient's characteristics, calculated previously
# is_debug:               Whether or not statements are printed over time.
# 
# and returns 1 value:
# 
# qlearning_dataset_mod: The full training dataset configured to be converted into states and actions
# all_lower_ranges:      The lower bound for the reward values
# all_upper_ranges:      The upper bound for the reward values
"""
def construct_prestate_matrix_train(train_90d:pd.DataFrame, train_blocs:pd.DataFrame, closest_clusters:List[List[int]],
                                    train_chosen_actions:pd.DataFrame, train_id_list:pd.DataFrame, 
                                    is_debug:bool = True, action_count:int = 20, 
                                    state_count:int = 750) -> (pd.DataFrame, List[int], List[int]):
    ###
    # BEGIN CONSTRUCTION OF PRE-STATE MATRIX
    # This will be used to build the full state/action matrix
    ### 

    # Based on whether or not a patient is dead, we establish the range of possible values:
    # If they have died, the range is [-100, 100]
    # If they are alive, the range is [100, -100]
    range_vals:List[int] = [100, -100]
    # Convert the range of values for a patient's status (dead or alive) from 0 or 1 to -1 or 1
    # This will enable ranges to suit the above criteria [-100, 100] or [100, -100]
    train_90d_polarity:List[int] = (2 * (1 - train_90d) - 1)
    range_matrix:List[int] = [np.multiply(polarity, range_vals) for polarity in train_90d_polarity]
    # Grab the lower range limit and upper range limit seperately in order to build the
    # full range of reward values
    all_lower_ranges:List[int] = [i[0] for i in range_matrix]
    all_upper_ranges:List[int] = [i[1] for i in range_matrix]
        
    # The qlearning_dataset prior to modification contains 6 columns and ~190885 rows (around 75% of the data)
    # The columns are as follows:
    #
    # training_bloc: time_series stamps for a patient's state over time, very in range from {1..?}
    #
    # closest_cluster_index: The index of the nearest cluster to the z-scores of the patient's data, 
    # corresponding actual data for each cluster's index (i) can be found in cluster_values[i]
    #
    # chosen_action_index: The chosen action or representation of a patient's IV_Fluid and Vasopressor status [0 - 24]
    # 
    # 90d_mortality_status: 0 means the patient is alive 90 days after discharge from ICU
    #                      1 means the patient is dead  90 days after discharge from ICU
    #
    # lower_range + upper_range: An index to be used later on, gathered from the range index
    if is_debug:
        print("Training Blocs Length: ", str(len(train_blocs)), "\nClosest Clusters Length: ", str(len(closest_clusters[0])), 
              "\nAction List Length: ", str(len(train_chosen_actions)), "\nTrain 90d Length", str(len(train_90d)), 
              "\nRange Matrix Length: ", len(range_matrix), "\nTrain IDs Length", str(len(train_id_list)))
    qlearning_dataset:pd.DataFrame = pd.concat([pd.Series(train_blocs.tolist()), 
                                   pd.Series(closest_clusters[0]), 
                                   pd.Series(train_chosen_actions.tolist()), 
                                   pd.Series(all_lower_ranges),
                                   pd.Series(train_90d.tolist()),
                                   train_id_list], 
                                   axis=1, sort=False)
    qlearning_dataset.columns = ['training_bloc', 'closest_cluster_index', 'chosen_action_index', 'reward_value', '90d_mortality_status', 'patient_id']
    if is_debug: 
        print(qlearning_dataset)
    # Modify the set for the final time in order to construct the final life + death states for each patient
    qlearning_dataset_mod = modify_qlearning_dataset(ql_dataset = qlearning_dataset, state_count = state_count)
    if is_debug:
        # Total patients being observed in the test
        print(len(qlearning_dataset_mod[qlearning_dataset_mod['training_bloc'] == 1]['training_bloc']))
        # Show that we now have end states established 
        print(len(qlearning_dataset[qlearning_dataset['chosen_action_index'] == action_count]['chosen_action_index']))
        print(len(qlearning_dataset_mod[qlearning_dataset_mod['chosen_action_index'] == action_count]['chosen_action_index']))
        print(len(qlearning_dataset_mod[qlearning_dataset_mod['closest_cluster_index'] == state_count]['chosen_action_index']))
        print(len(qlearning_dataset_mod[qlearning_dataset_mod['closest_cluster_index'] ==  state_count + 1]['chosen_action_index']))
        print(qlearning_dataset_mod, "\n")
    return qlearning_dataset_mod, all_lower_ranges, all_upper_ranges
    


# In[59]:


"""
# modify_qlearning_dataset is a function that takes a dataframe intended for qlearning and modifies
# it in preparation for the ML. In essence, it runs through creating the life and and death states in preparation
# for constructing the MDP
#
# Parameters: 
# ql_dataset:  The Dataset we want to modify in preparation
# is_training: Whether we are building the training set or not.
# is_debug:    Whether prints are included or not
#       
# 
# Returns: qlearning_dataset_mod - The modified dataset
"""

def modify_qlearning_dataset(ql_dataset:pd.DataFrame, state_count:int) -> pd.DataFrame:
    # The base qlearning_dataset does not account for endpoints in either life or death
    # These states have not been established yet, which is what this step corrects
    qlearning_dataset:pd.DataFrame = ql_dataset.copy()
    qlearning_dataset_len:int = len(qlearning_dataset.index)
    # We need space to add a death/life state for every patient, about a 20% increase in size from the original MDP
    # We will cut the excess off by the end of the loop
    qlearning_dataset_len_mod:float = int(np.floor(qlearning_dataset_len * 1.2))
    qlearning_dataset_mod:np.ndarray = []
    qlearning_dataset_mod:np.ndarray = np.array([[0 for i in range(0, 6)] for i in range(0, qlearning_dataset_len_mod)])
    # Start construction of modified data
    row:int = 0
    # In Markov theory, an absorbing state is one which can be entered, but cannot be left. (Similar to the Hotel California)
    # In the case of this experiment, those states are either life (state_count) or death (state_count + 1) per patient as
    # defined by me (750, 751)
    absorbing_states:List[int] = [state_count, state_count + 1]
    # Start the loop to begin capping the markov chain off at life and death states
    for i in range(0, qlearning_dataset_len - 1):
        # Use the already gathered data for each row
        qlearning_dataset_mod[row, :] = qlearning_dataset.iloc[i][0:6]
        # If we arrive at the terminal point (end of patient data), we need to point the MDP to either the death or life state
        if qlearning_dataset.iloc[i + 1]['training_bloc'] <= qlearning_dataset.iloc[i]['training_bloc']:
            # Grab the row
            whole_row:pd.DataFrame = qlearning_dataset.iloc[i]
            # Set most of the row to the original data's values, except set the action to be either state 750 or 751
            # Life or death respectively
            row = row + 1
            # We need bloc number, final state (life or death, 750 or 751), end action (-1), and the reward value (lower_range)
            qlearning_dataset_mod[row, :] = [whole_row['training_bloc'] + 1, absorbing_states[whole_row['90d_mortality_status']], -1,  whole_row['reward_value'], whole_row['90d_mortality_status'], whole_row['patient_id']]
        row = row + 1
    # Add in the last row
    whole_row:pd.DataFrame = qlearning_dataset.iloc[len(qlearning_dataset.index) - 1]
    qlearning_dataset_mod[row, :] = [whole_row['training_bloc'] + 1, absorbing_states[whole_row['90d_mortality_status']], -1,  whole_row['reward_value'], whole_row['90d_mortality_status'], whole_row['patient_id']]
    
    row = row + 1
    # Get rid of the unneeded rows
    qlearning_dataset_mod:pd.DataFrame = pd.DataFrame(qlearning_dataset_mod[0:row, :])
    qlearning_dataset_mod.columns = ['training_bloc', 'closest_cluster_index', 'chosen_action_index', 'reward_value', 'death_state', 'patient_id']
    # Set all rows not in the terminal states to 0 reward to start
    qlearning_dataset_mod.loc[qlearning_dataset_mod['chosen_action_index'] != -1,'reward_value'] = 0
    return qlearning_dataset_mod


# In[60]:


###
#  Now that we officially have some a valid bloc for actions, and a valid set of states, it's time 
#  to begin building the transitions matrix.
###

### If the matrix is bidirectional (S1 -> S2, S2 -> S1 are both valid, we can build two matrices)

### 
# The MDP Toolbox we are going to be using requires Transition and Reward Matrices to be in the form
# M(action, state1, state2)
###

"""
# The create_transition_matrix method takes 4 arguments:
# num_actions: The total number of possible actions (calculated by action_count ^ 2 or in py, action_count ** 2)
# num_states:  Number of states the model uses
# qlearning_dataset: The dataset that will be used for the qlearning process
# transition_threshold: How many actions do we want to deem as scarce and not worth keeping (default = 5)
# reverse: If false, the matrix that is created is represented as transition[A][S1][S2], if true: transition[A][S2][S1]
# 
# and returns 2 values:
# transition_matrix: The counts of which actions were chosen in which states
# physician_policy:  The transition_matrix that has been turned into probabilties by dividing counts in each state by 
# total counts
# 
"""
def create_transition_matrix(num_actions:int, num_states:int, ql_data_input:pd.DataFrame, 
                             transition_threshold:int = 5, reverse:bool = False) -> (int_matrix2D, float_matrix2D):
    # The transition matrix is a 3D construct, involving a transition between two states
    # and an action. The dimensions for the matrix are (state_count * 2) * (state_count + 2) * action_count
    transition_matrix:float_matrix2D = [[[0 for i in range(0, num_states + 2)] for i in range(0, num_states + 2)] for i in range(0, num_actions)]
    # NP Arrays allow for more compact and efficient slicing
    transition_matrix = np.array(transition_matrix).astype(float)
    # We also need a matrix to denote the policy that corresponds with taken a particular action from a state
    transition_policy_count:int_matrix2D = [[0 for i in range(0, num_states + 2 )] for i in range(0, num_actions)]
    transition_policy_count = np.array(transition_policy_count).astype(float)
    # Iterate over the actual data in order to form the actual states and their corresponding actions
    # As soon as we hit the next patient (the next row has a training bloc value of 1), we stop processing actions for that patient
    for i in range(0, len(ql_data_input) - 1):
        # Since 1 is our 'endpoint' for each patient, there are no actions we can take from this point on
        if ql_data_input.iloc[i + 1]['training_bloc'] > ql_data_input.iloc[i]['training_bloc']:
            S1:int = ql_data_input.iloc[i]['closest_cluster_index']
            S2:int = ql_data_input.iloc[i + 1]['closest_cluster_index'] 
            action_id:int = ql_data_input.iloc[i]['chosen_action_index']
            if not(reverse):
                # Count the number of times S1 -> S2 is taken using action A
                transition_matrix[action_id][S1][S2] = transition_matrix[action_id][S1][S2] + 1
            else:
                # Count the number of times S1 -> S2 is taken using action A
                transition_matrix[action_id][S2][S1] = transition_matrix[action_id][S2][S1] + 1
                
            # Count the number of times action A is used to transition from S1
            transition_policy_count[action_id][S1] = transition_policy_count[action_id][S1] + 1        

    # In order to avoid drastically altering our model, we fix a constant
    # value (set by default to 5), in order to declare sparse actions 
    # as essentially not happening (make their count 0)
    for i in range(0, num_actions):
        for j in range(0, num_states + 2):
            if transition_policy_count[i][j] <= transition_threshold:
                transition_policy_count[i][j] = 0 
    # Now, we want to prevent transitions from state -> state using
    # a certain action if that action is sparse or nonexistant
    for i in range(0, num_actions):
        for j in range(0, num_states + 2):
            if not(reverse):
                # Declare the weight of an unachievable action to have a zero probability
                if transition_policy_count[i][j] == 0:
                    transition_matrix[i,j,:] = 0
                    # All probabilities must be declared, even unreachable states, an easy work around 
                    # to this issue is to simply declare the same state to have a probability of 1
                    # https://stackoverflow.com/questions/43665797/must-a-transition-matrix-from-a-markov-decision-process-be-stochastic
                    transition_matrix[i,j,j] = 1
                # This weights the MDP based on the probability of taking one action from a state
                # As opposed to taking any other possible action from that state
                # S1 -> S2 might be 50%, S1 -> S3 20%, and S1 -> S4 30%
                else:
                    transition_matrix[i,j,:] = transition_matrix[i,j,:]/np.float64(transition_policy_count[i][j])
            else:
                # Declare the weight of an unachievable action to have a zero probability
                if transition_policy_count[i][j] == 0:
                    transition_matrix[i,:,j] = 0
                    # All probabilities must be declared, even unreachable states, an easy work around 
                    # to this issue is to simply declare the same state to have a probability of 1
                    # https://stackoverflow.com/questions/43665797/must-a-transition-matrix-from-a-markov-decision-process-be-stochastic
                    transition_matrix[i,j,j] = 1
                # This weights the MDP based on the probability of taking one action from a state
                # As opposed to taking any other possible action from that state
                # S1 -> S2 might be 50%, S1 -> S3 20%, and S1 -> S4 30%
                else:
                    transition_matrix[i,:,j] = transition_matrix[i,:,j]/np.float64(transition_policy_count[i][j])
    
    # Ensure no divisions create NaNs or infinities
    transition_matrix = np.nan_to_num(transition_matrix)
    # Determine the phyisican's policy based on total count
    # This comes in handy later when comparing model ability
    total_transitions:float = sum(transition_policy_count)
    physician_policy:float_matrix2D = np.divide(transition_policy_count, total_transitions)
    return transition_matrix, physician_policy


# In[61]:


"""
eval_predict_correct checks the current reward value against whether or not
the patient's terminal state was predicted correctly and evaluates whether or
not the model should be rewarded or penalized

Input: 
reward_value - The total reward value evaluated for a given patient
correct_answer - 0 if the patient lived, 1 if the patient died

Output:
reponse_reward - The total amount of reward that justifies a particular response
"""
def eval_predict_correct(reward_value:int, correct_answer:int):
    # Values to help clean up readability
    PATIENT_LIVED = 0
    PATIENT_DIED = 1
    PREDICTED_DEAD = reward_value < 0
    PREDICTED_LIVING = reward_value >= 0
    return_reward = 0
    # If the model guessed correctly and the patient died, weight the path -100 
    if (PREDICTED_DEAD and correct_answer == PATIENT_DIED):
        return_reward = -200
    # If the model guessed correctly and the patient lived, weight the path +100
    elif (PREDICTED_LIVING and correct_answer == PATIENT_LIVED):
        return_reward = 100
    # If the model guessed incorrectly and the patient lived, weight the path +100
    elif (PREDICTED_DEAD and correct_answer == PATIENT_LIVED):
        return_reward = 100
    # If the model guessed incorrectly and the patient died, weight the path -100
    elif (PREDICTED_LIVING and correct_answer == PATIENT_DIED):
        return_reward = -200
    # This should not be reachable with a properly prepared dataset
    else:
        print("Your dataset is not prepared properly")
    return return_reward
    


# In[62]:


"""
#
#
#
#
#
#
"""
def modify_incoming_data(ql_train_set_Q:pd.DataFrame, data_option:str):
    patient_indexes:pd.DataFrame = ql_train_set_Q[['patient_id', 'death_state']].drop_duplicates()
    all_living:pd.DataFrame = patient_indexes[patient_indexes['death_state'] == 0]
    all_dead:pd.DataFrame = patient_indexes[patient_indexes['death_state'] == 1]
    if data_option == "normal":
        pass
    elif data_option == "striped":
        # Striped output alternates living and dead patients
        all_living_list:List[str] = all_living['patient_id'].tolist()
        all_dead_list:List[str] = all_living['patient_id'].tolist()
        striped_list:List[str] = []
        l_len = len(all_living_list)
        d_len = len(all_dead_list) 
        if l_len == d_len:
            for i in range(0, l_len):
                striped_list.append(all_dead_list[i])
                striped_list.append(all_living_list[i])
        elif l_len > d_len:
            for i in range(0, d_len):   
                striped_list.append(all_dead_list[i])
                striped_list.append(all_living_list[i])
            striped_list.extend(all_living_list[d_len:])
        elif d_len > l_len:
            for i in range(0, l_len):
                striped_list.append(all_dead_list[i])
                striped_list.append(all_living_list[i])
            striped_list.extend(all_dead_list[d_len:])
        return striped_list
    elif data_option == "all_dead_first":
        patient_indexes = all_dead.append(all_living, ignore_index=True)
    elif data_option == "all_living_first":
        patient_indexes = all_living.append(all_dead, ignore_index=True)
    else:
        pass
    return patient_indexes['patient_id'].tolist()


# In[63]:


"""
# offpolicy_Q_learning_eval is a method that takes 6 arguments and returns 2 items
# This method evaluates the performance of the MDP determined previously
# 
# Parameters:
# ql_train_set_Q: The actual dataset that serves as our proto-MDP
# phys_pol: A 2D (actions X states) matrix that shows what the phyisican chose according dataset probabilities
# gamma: A hyperparameter for determining how much we value previous data
# alpha: A hyperparameter that weights our reward function at each step
# numtraces: Number of Q-Learning iterations we would like to perform
# num_actions: Total number of actions in the set (For Sepsis: 25)
# num_clusters: Total number of states in the set (For Sepsis: 752)
# is_training:  If False, this is the first phase (construction of the model). If True, the model is being Trained
# 
# Returns:
# Q_Equation = The set of Q-Values obtained by the algorithm's performance
# sum_Q_values = The Q-Equation's performance at a given step in the algorithm
"""
def offpolicy_Q_learning_eval(ql_train_set_Q: pd.DataFrame, gamma: float, alpha: float, 
                              numtraces: int, num_actions: int, 
                              num_clusters: int, stopping_difference: float, 
                              is_random=False, data_option="normal",
                              modulus_val:int=5000) -> (float_matrix2D, List[float]):
    # We need to save the Q-value for each run 
    sum_Q_values:List[float] = np.zeros((numtraces))
    # Where the Q-Values are saved at each given run
    Q_Equation:int_matrix2D = np.zeros((num_actions, num_clusters))
    # We need to save the Average Q value after so many iterations
    previous_avg_Q:int = 0
    # Grab the full list of IDs, using the desired input modifier
    patient_indexes:List[str] = modify_incoming_data(ql_train_set_Q=ql_train_set_Q, data_option=data_option)
    # This is a seperate index used for the Sum of the Q-Values
    jj:int = 0
    # If we don't want to use a random set of the data points, only do each data point once
    if not(is_random):
        numtraces = len(patient_indexes)
    # We iterate for the total number of times we want to do this process (random) or all the ids
    for i in range(0, numtraces):
        # Either go in order, or randomly choose a patient
        patient_id:int = 0
        if is_random:
            patient_id = random.choice(patient_indexes)
        else:
            patient_id = patient_indexes[i]
        # As Q-learning progreses, we need a data structure to track progress
        full_trace:List[Tuple[float, int, int]] = []
        # Stash a total reward value for the whole run
        total_reward:int = 0
        # Evaluate over a single patient
        single_patient = ql_train_set_Q[ql_train_set_Q['patient_id'] == patient_id]
        # Iterate over a single patient's data to contribute to the overall Q_Equation
        for i in range(0, len(single_patient.index) - 1):
            # Grab state (Initial State at this point)
            state_index:int = single_patient.iloc[i + 1]['closest_cluster_index']
            # Grab action taken from this point
            action_index:int = single_patient.iloc[i + 1]['chosen_action_index']
            # Grab reward provided by taken an action from this state to the next
            reward_value:float = single_patient.iloc[i + 1]['reward_value']
            # Add to the total
            total_reward = total_reward + reward_value
            # A 'step' in the trace, a single data point snapshot
            trace_step:Tuple[float, int, int] = (reward_value, state_index, action_index)
            # Add the step to the full trace
            full_trace.append(trace_step)
        # Grab the full length of the trace
        trace_length:int = len(full_trace)
        # Grab whether the patient lived or died
        death_state:int = single_patient.iloc[0]['death_state']
        return_reward:float = eval_predict_correct(total_reward, death_state)
        # Walk the trace stack backwards to construct the Q-Equation
        # The terminal state and the penulitmate state both should not have a reward
        for j in range(trace_length - 2, - 1, -1):
            # Grab the state, action, and reward at each step
            step_state:int = full_trace[j][1]
            step_action:int = full_trace[j][2]
            # Use alpha blending to blend a portion of the old reward with the new reward
            Q_Equation[step_action, step_state] = (1 - alpha) * Q_Equation[step_action, step_state] + alpha * return_reward
            # Cap the range for node values (-100, 100)
            if Q_Equation[step_action, step_state] > 100:
                Q_Equation[step_action, step_state] = 100
            if Q_Equation[step_action, step_state] < -100:
                Q_Equation[step_action, step_state] = -100
            # Recall we have a gamma value to determine the impact of previous decisions on future ones
            # Note: this is a Hyperparameter (a parameter on the model itself)
            return_reward = return_reward * gamma  + full_trace[j][0]
            # Save the overall value based on the current states and actions avaiable at the 
            # current iteration
        sum_Q_values[jj] = np.sum(Q_Equation)
        jj = jj + 1
        # If we haven't hit our max iterations, we still want to see if we should keep pushing forward
        # If there is no noticable progress, we want to stop
        # This is only applicable if we are not using all the training data set patients
        if is_random:
            # Perform a check every modulus_val runs
            if i % modulus_val == 0:
                # Grab the current slice of unchecked {modulus_val} values
                slice_mean:float = np.mean(sum_Q_values[j - modulus_val:j])
                # Calculate the difference between current and last average 
                max_difference:float =(slice_mean - previous_avg_Q)/previous_avg_Q
                # Check if the average of this {modulus_val} values is less than 0.001 away from the previous
                if abs(max_difference) < stopping_difference:
                    break
                previous_avg_Q = slice_mean
    # Trim off the portion of the list we did not use
    sum_Q_values = sum_Q_values[0:jj]
    return Q_Equation, sum_Q_values


# In[64]:


"""
construct_trained_model takes an existing dataframe and information regarding the Q_Equation
already constructing and modifies it with correct Reward Values

Input:
ql_final_dataset: The dataset that is inputted into the Q-Learning algorithm
state_count: The number of states built by the K-Means algorithm
total_actions: The number of actions constructed in the action matrix
physician_policy: Probability distributions for likelihood of patient transitions from state to state
Q_Equation: The reward values generated for transitions from state to state using a given action generated
by the Q-Learning algorithm

Output: 
qlearning_train_final: The finalized dataframe with reward values inserted
"""
def construct_trained_model(ql_final_dataset:pd.DataFrame, state_count:int, 
                            total_actions:int, weighted_probabilities:bool, 
                            physician_policy:List[List[float]], Q_Equation:List[List[float]]) -> pd.DataFrame:
    # Set variables for readability
    NO_ACTION = -1
    # Make a final duplicate of the data
    qlearning_train_final:pd.DataFrame = ql_final_dataset.copy()
    # If weights are considered, use the weighted rewards from the Q_Equation, otherwise 
    # use the Q_Equation as is 
    if weighted_probabilities:
        # Weight Q-Value rewards according to their frequency of occuring
        # The Q-Equation for a given state + action pair is equivalent to the reward value
        # The Phyiscian Policy is the probability of that action ocurring given a state
        # These weights prevent rare events from having massively scewed rewards
        # For example, a path that occurs through a given state exactly once would have a much larger
        # reward than a frequently traveled path, which could scew the data.
        value_matrix = np.zeros((state_count, total_actions))
        for i in range(0, state_count):
            for j in range(0, total_actions):
                value_matrix[i][j] = physician_policy[j][i] * Q_Equation[j][i]
        for i in range(0, len(qlearning_train_final.index)):
            row = qlearning_train_final.iloc[i]
            if((row['closest_cluster_index'] != state_count) and (row['closest_cluster_index'] != state_count + 1)):
                row['reward_value'] = value_matrix[row['closest_cluster_index']][row['chosen_action_index']]
    else:
        for i in range(0, len(qlearning_train_final.index) - 1):
            # The next state and action pair has a reward value that represents the reward
            # for the next transition
            curr_row = qlearning_train_final.iloc[i]
            next_row = qlearning_train_final.iloc[i + 1]
            # If we are not going into the terminal state or at the terminal state, assign reward
            if((curr_row['chosen_action_index'] != NO_ACTION) and (next_row['chosen_action_index'] != NO_ACTION)):
                curr_row['reward_value'] = Q_Equation[next_row['chosen_action_index']][next_row['closest_cluster_index']]
    
    return qlearning_train_final
        


# In[65]:


"""
# evaluate_model_accuracy calculates summations for all patients and determines if the overall reward value
# predicts a patient to live or die 
# 
# Input: 
# qlearning_train_final: The complete dataframe with reward values intact
# test_name: The desired title to be printed to the console
# is_debug: Whether or not to include print statements
# gamma - A hyperparameter for determining how much we value previous data
# alpha - A hyperparameter that weights our reward function at each step
# run_num - The number of the testing iteration the model is on 
# 
# Output:
# N/A
# 
"""
def evaluate_model_accuracy(qlearning_train_final:pd.DataFrame, test_name:str, is_debug:bool, 
                            alpha:float, gamma:float, run_num:int):
    all_patients_ids = qlearning_train_final['patient_id'].drop_duplicates().tolist()
    # Iterate until the full set is done
    total_patients:int = len(all_patients_ids)
    unique_total:pd.DataFrame = qlearning_train_final[['patient_id', 'death_state']].drop_duplicates()
    total_alive:int = unique_total['death_state'].value_counts()[0]
    total_dead:int = unique_total['death_state'].value_counts()[1]
    correct_guesses:int = 0
    dead_instead_live:int = 0
    live_instead_dead:int = 0
    zero_valued_reward:int = 0
    bad_average_dead:List[float] = []
    bad_average_alive:List[float] = []
    good_average_dead:List[float] = []
    good_average_alive:List[float] = []
    for id_val in all_patients_ids:
        all_rewards:pd.Series = qlearning_train_final[qlearning_train_final['patient_id'] == id_val]['reward_value']
        # We are not supposed to know the terminal reward for evaluation, this is the life/death state
        all_rewards:pd.Series = all_rewards[:-1]
        # Sum the total reward value for a given column, determine the sign value
        reward_sum:float = np.sum(all_rewards)
        reward_sign:int = np.sign(np.sum(all_rewards))
        # Treat 0s as Living
        if reward_sign == 0:
            zero_valued_reward = zero_valued_reward + 1
            reward_sign = 1
        # Grab the death state of the patient
        death_state = qlearning_train_final[qlearning_train_final['patient_id'] == id_val]['death_state'].tolist()[0]
        # Change Values to match reward system 0 -> +1 -> Patient Lived, 1 -> -1 -> Patient Died
        PATIENT_LIVED = 0
        PATIENT_DIED = 1
        if death_state == PATIENT_LIVED:
            death_state = 1
        else:
            death_state = -1
        # If the prediction and actual values matched, the model predicted correctly
        if reward_sign == death_state:
            if reward_sign == -1:
                good_average_dead.append(reward_sum)
            else:
                good_average_alive.append(reward_sum)
            correct_guesses = correct_guesses + 1
        # If the patient was presumed to be dead and lived, count it
        elif reward_sign == -1:
            bad_average_alive.append(reward_sum)
            live_instead_dead = live_instead_dead + 1
        # If the patient was presumed to be alive and died, count it
        else:
            bad_average_dead.append(reward_sum)
            dead_instead_live = dead_instead_live + 1
    # Grab all the calculated values
    overall_accuracy:float = correct_guesses/total_patients
    dead_accuracy:float = (total_dead - dead_instead_live) / total_dead
    live_accuracy:float = (total_alive - live_instead_dead) / total_alive
    if is_debug:
        print("Test Name: " + test_name)
        print("Overall Accuracy: " + str(overall_accuracy))
        print("Accuracy for Dead: " + str(dead_accuracy))
        print("Accuracy for Living: " + str(live_accuracy))
        print("Living People Guessed Dead: " + str(live_instead_dead))
        print("Dead People Guessed Living: " + str(dead_instead_live))
        print("Total Guesses: " + str(total_patients))
        print("Correct Guesses: " + str(correct_guesses))
        print("Alive People: " + str(total_alive))
        print("Dead People: " + str(total_dead))
        print("Empty Paths: " + str(zero_valued_reward))
        print("Average Incorrect Guessed Dead Reward: " + str(np.average(bad_average_alive)))
        print("Average Incorrect Guessed Alive Reward: " + str(np.average(bad_average_dead)))
        print("Average Correct Guessed Dead Reward: " + str(np.average(good_average_dead)))
        print("Average Correct Guessed Alive Reward: " + str(np.average(good_average_alive)))
        print("\n")
    # Construct a formatted CSV string with all the values that have been set
    total_string = (f'{run_num},{alpha},{gamma},{total_patients},{correct_guesses},'
                    f'{overall_accuracy},{dead_accuracy}'
                    f'{live_accuracy},{live_instead_dead},{dead_instead_live},'
                    f'{total_alive},{total_dead},{zero_valued_reward},'
                    f'{test_name}')
    # Check the test name and write to the corresponding file
    if test_name == 'Test_Weighted':
        with open('test_weighted_runs.csv', 'a') as f:
            f.write(total_string + "\n")
    elif test_name == 'Test_No_Weighting':
        with open('test_no_weighting_runs.csv', 'a') as f:
            f.write(total_string + "\n")
    elif test_name == 'Train_Weighted':
        with open('train_weighted_runs.csv', 'a') as f:
            f.write(total_string + "\n")
    else:
        with open('train_no_weighting_runs.csv', 'a') as f:
            f.write(total_string + "\n")
    # Return the final formatted string if desired
    return total_string


# In[66]:


"""
# PCA (Principal Component Analysis) is a technique to perform dimensionality reduction in large featured datasets (I.E) MIMIC
# It turns a series of features into multi-component lists, each possessing a certain variance amount 
# based on the inputted dataset
# 
# Input:
# MIMIC_zscores - The MIMIC dataset zscores in the form of a dataframe
# variance_allowed - The decimal representation of how much of the original variance desired to be intact
# a value of 0 will have a completely unique dataset, 1 will be the original dataset in components
# is_debug: Enables print statements
# num_components: For the initial run, this should be set to -1, the following runs should be set to the output
# of this function. This value is the number of components that are presented in each list PCA generates
#
# Output: 
# pca_dataset: The PCA generated dataset
# num_components: The optimal number of components selected by PCA with the specified variance_allowed
# 
"""
def pca_variance_analysis(MIMIC_zscores:pd.DataFrame, variance_allowed:float, is_debug:bool, num_components:int) -> float_matrix2D:
    if num_components == -1:
        model = PCA()
        model.fit(MIMIC_zscores.values)
        total_variances = np.cumsum(model.explained_variance_ratio_)
        num_components = 0
        for variance in total_variances:
            if variance < variance_allowed:
                num_components = num_components + 1
        if is_debug: 
            print("Number of Components Chosen: ", num_components)
    real_model = PCA(n_components=num_components)
    pca_dataset = real_model.fit_transform(MIMIC_zscores.values)
    return pca_dataset, num_components
    


# In[67]:


"""
# aggregate_dataset_construct is a function that groups all the functions to build the initial dataset for training
#
# Input: 
# debug_flag - Whether or not print statements are included
# save_to_file - Whether or not intermediate steps are saved to file
# regenerate_flag - Whether to run clustering, or read from file
# regenerate_cross - Whether or not to run cross_validation split or read from file
#
# Output:
# train_flag_set - All the rows marked to be used for training data
# test_flag_set - All the rows mared to be used for testing
# MIMIC_zscores - The MIMIC data reduced to Z-Scores
"""
def aggregate_dataset_construct(debug_flag:bool = False, save_to_file:bool = True, 
                                regenerate_flag:bool = False, regenerate_cross:bool=False):
    # Extract the initial columns and build the first dataset from the CSV file
    colbin, colnorm, collog, MIMIC_raw, id_count, icu_ids = extract_init_column_data(patient_data=patientdata, debug_flag=False)
    # Construct the Z-Scores version of the dataframe
    MIMIC_zscores:pd.DataFrame = construct_zscores(colbin, colnorm, collog, MIMIC_raw, debug_flag=False) 
    # Allow for dynamic construction of chosen factors
    all_factors = [colbin, colnorm, collog]
    # Trick to Flatten nested list
    all_factors = [item for sublist in all_factors for item in sublist]
    # Stratify the train and test sets
    train_ids_set, test_ids_set, train_flag_set, test_flag_set = cross_validate_and_balance(icu_ids, debug_flag, 
                                                                             save_to_file, regenerate_flag, 
                                                                             patientdata, regenerate_cross,
                                                                             all_factors)
    # The flags are all we need for the subsequent steps
    return train_flag_set, test_flag_set, MIMIC_zscores
    


# In[68]:


"""
# aggregate_training_clustering is a function that groups all the functions to cluster and build the training dataset
#
# Input: 
# train_flag - All the rows marked to be used for training
# test_flag - All the rows marked to be used for testing
# MIMIC_zscores - The Z-Scores of the standard patientdata set
# is_debug - Whether or not to use print statements
# save_flag - Whether or not intermediate steps are saved to files
# variance_allowed - Tolerance used for PCA if it is enabled (Principal Component Analysis)
# pca_flag - Whether or not to enable PCA
# graph_flag - Whether or not the graph of actions should be displayed
# load_zscores - Whether or not zscores should be loaded from file
# regenerate_flag - Whether or not to rerun clustering 
# 
# Output:
# qlearning_dataset_train_final - The final, modified, clustered dataset for training 
# qlearning_dataset_test_final - The final, modified, clustered dataset for testing
# transition_mat - The counts of transitioning from one state to another using a given action
# physician_policy - The probabilities of transitioning from one state to another using a given action
# total_actions - Total number of actions to be used for training
# state_count - Total number of states to be used for training
# 
"""
def aggregate_training_clustering(train_flag:pd.DataFrame, test_flag:pd.DataFrame, MIMIC_zscores:pd.DataFrame,
                                  is_debug:bool=False, save_flag:bool=True, variance_allowed=0.8, 
                                  pca_flag:bool = False, graph_flag:bool = False, 
                                  load_zscores:bool=False, regenerate_flag=True):
    # Construct the training and testing variants of the Z-Scores
    train_zscores, test_zscores, train_blocs, test_blocs, train_id_list, test_id_list, train_90d, test_90d =     zscores_for_train_and_test(train_flag, test_flag, MIMIC_zscores, debug_flag=is_debug, save_flag=save_flag, 
                               patient_data=patientdata, bloc_name='bloc', id_name='icustayid', death_name='mortality_90d')
    # If it desired to perform Principal Component Analysis (PCA) instead of using raw Z-Scores
    if pca_flag:
        train_zscores, num_train_comp = pca_variance_analysis(MIMIC_zscores=train_zscores, 
                                                       variance_allowed=variance_allowed, is_debug=is_debug, num_components=-1)
        test_zscores, _ = pca_variance_analysis(MIMIC_zscores=test_zscores, variance_allowed=variance_allowed, 
                                                 is_debug=is_debug, num_components=num_train_comp)
    # Optimize or load in the optimal number of clusters (for this dataset, that cluster count happens to be 42)
    optimal_cluster_count = calculate_optimal_clusters_driver(data_set=train_zscores, run_optimal_clusters=False, 
                                                              run_multithread=False, show_graph=graph_flag, 
                                                              thread_count=-1, max_state_count=15)
    # Hard override for testing
    optimal_cluster_count = 10
    # State count is the optimal cluster count 
    state_count = optimal_cluster_count
    # Perform K-Means clustering with the optimal number of clusters
    cluster_values, cluster_labels, train_zscores, closest_clusters, closest_clusters_test =     kmeans_cluster_calculations(train_zscores=train_zscores, test_zscores=test_zscores,
                                max_num_iter=32, state_count=optimal_cluster_count, num_loops_per_iter=10000,
                                debug_flag=is_debug, regenerate_flag=regenerate_flag, load_zscores=load_zscores)
    # Training and Testing Construction based on cluster outputs
    train_chosen_actions, test_chosen_actions, action_values_matrix, action_list, action_count = build_dataset_actions(patientdata=patientdata, 
                      first_column="SOFA", second_column="qSOFA", 
                      num_groups_first_column=5, num_groups_second_column=4, 
                      debug_flag=is_debug, graph_flag=graph_flag, train_flag=train_flag, test_flag=test_flag)
    # Construct the State->Action Matrix for the training set
    qlearning_dataset_train_final, _, _ = construct_prestate_matrix_train(train_90d = train_90d, 
                                                                                            train_blocs = train_blocs, 
                                                                                            closest_clusters = closest_clusters,
                                                                                            train_chosen_actions = train_chosen_actions, 
                                                                                            train_id_list = train_id_list,
                                                                                            is_debug=is_debug,
                                                                                            action_count=action_count, 
                                                                                            state_count=state_count)
    qlearning_dataset_test_final, _, _ = construct_prestate_matrix_train(train_90d = test_90d, 
                                                                     train_blocs = test_blocs, 
                                                                     closest_clusters = closest_clusters_test,
                                                                     train_chosen_actions = test_chosen_actions,
                                                                     train_id_list = test_id_list,
                                                                     is_debug=False,
                                                                     action_count=action_count, 
                                                                     state_count=state_count)
    # Constructing Transition Matrix(A, State1, State2)
    total_actions:int = len(action_values_matrix)     
    # Execute the function call
    transition_mat, physician_policy = create_transition_matrix(num_actions = total_actions, 
                                                                num_states = state_count,ql_data_input = 
                                                                qlearning_dataset_train_final, 
                                                                transition_threshold = 5, 
                                                                reverse = False)
    return qlearning_dataset_train_final, qlearning_dataset_test_final, transition_mat,           physician_policy, total_actions, state_count
    


# In[69]:


"""
# aggregate_Q_learning_model_construction is a function that groups all the functions to perform Q-Learning
#
# Input: 
# qlearning_dataset_train_final - The final, modified, clustered dataset for training 
# qlearning_dataset_test_final - The final, modified, clustered dataset for testing
# gamma - A hyperparameter for determining how much we value previous data
# alpha - A hyperparameter that weights our reward function at each step
# state_count - Total number of states to be used for training
# total_actions - Total number of actions to be used for training
# physician_policy - The probabilities of transitioning from one state to another using a given action
#
# Output:
# qlearning_dataset_train_final_weighted - Q-Learning final model output for the weighted training variant
# qlearning_dataset_train_final_no_weighting - Q-Learning final model output for the unweighted training variant
# qlearning_dataset_test_final_weighted - Q-Learning final model output for the weighted test set 
# qlearning_dataset_test_final_no_weighting - Q-Learning final model output for the unweighted test set
# 
"""
def aggregate_Q_learning_model_construction(qlearning_dataset_train_final:pd.DataFrame, 
                                            qlearning_dataset_test_final:pd.DataFrame,
                                            gamma:float, alpha:float, state_count:int, 
                                            total_actions:int, physician_policy:pd.DataFrame):
    # Two rounds of Q-Learning, one to set initial values, and then one to tweak those
    Q_Equation, sum_Q_values = offpolicy_Q_learning_eval(
            ql_train_set_Q=qlearning_dataset_train_final,
            gamma=gamma, 
            alpha=alpha,
            numtraces=30000,
            num_actions=total_actions,
            num_clusters=state_count,
            stopping_difference=0.001,
            is_random=False,
            modulus_val=5000
    )
    # First Round Model
    qlearning_train_final = construct_trained_model(ql_final_dataset=qlearning_dataset_train_final, 
                                                state_count=state_count, total_actions=total_actions, 
                                                weighted_probabilities=True, physician_policy=physician_policy,
                                                                Q_Equation=Q_Equation)
    """
    Q_Equation, sum_Q_values = offpolicy_Q_learning_eval(
            ql_train_set_Q=qlearning_train_final,
            gamma=gamma, 
            alpha=alpha,
            numtraces=30000,
            num_actions=total_actions,
            num_clusters=state_count,
            stopping_difference=0.001,
            is_random=True,
            modulus_val=5000
    )
    """
    # Second Round Model
    qlearning_dataset_train_final_weighted = construct_trained_model(ql_final_dataset=qlearning_dataset_train_final, 
                                                state_count=state_count, total_actions=total_actions, 
                                                weighted_probabilities=True, physician_policy=physician_policy,
                                                                             Q_Equation=Q_Equation)
    qlearning_dataset_train_final_no_weighting = construct_trained_model(ql_final_dataset=qlearning_dataset_train_final, 
                                                state_count=state_count, total_actions=total_actions, 
                                                weighted_probabilities=False, physician_policy=physician_policy,
                                                                              Q_Equation=Q_Equation)
    qlearning_dataset_test_final_weighted = construct_trained_model(ql_final_dataset=qlearning_dataset_test_final, 
                                                state_count=state_count, total_actions=total_actions, 
                                                weighted_probabilities=True, physician_policy=physician_policy,
                                                                             Q_Equation=Q_Equation)
    qlearning_dataset_test_final_no_weighting = construct_trained_model(ql_final_dataset=qlearning_dataset_test_final, 
                                                state_count=state_count, total_actions=total_actions, 
                                                weighted_probabilities=False, physician_policy=physician_policy,
                                                                              Q_Equation=Q_Equation)
    return qlearning_dataset_train_final_weighted, qlearning_dataset_train_final_no_weighting,           qlearning_dataset_test_final_weighted, qlearning_dataset_test_final_no_weighting
    


# In[70]:


"""
# aggregate_Q_learning_evaluation is a function that evaluates all the model output and save it to an appended file
#
# Input: 
# qlearning_dataset_train_final_weighted - Q-Learning final model output for the weighted training variant
# qlearning_dataset_train_final_no_weighting - Q-Learning final model output for the unweighted training variant
# qlearning_dataset_test_final_weighted - Q-Learning final model output for the weighted test set 
# qlearning_dataset_test_final_no_weighting - Q-Learning final model output for the unweighted test set
#
# Output:
# N/A
# 
"""
def aggregate_Q_learning_evaluation(qlearning_dataset_train_final_weighted:pd.DataFrame, qlearning_dataset_train_final_no_weighting:pd.DataFrame,
                                    qlearning_dataset_test_final_weighted:pd.DataFrame, qlearning_dataset_test_final_no_weighting:pd.DataFrame,
                                    gamma:float, alpha:float, run_num:int):
    evaluate_model_accuracy(qlearning_dataset_test_final_weighted, test_name="Test_Weighted", is_debug=True, alpha=alpha, gamma=gamma, run_num=run_num)
    evaluate_model_accuracy(qlearning_dataset_test_final_no_weighting, test_name="Test_No_Weighting", is_debug=True, alpha=alpha, gamma=gamma, run_num=run_num)
    # evaluate_model_accuracy(qlearning_dataset_train_final_weighted, test_name="Train_Weighted", is_debug=True, alpha=alpha, gamma=gamma, run_num=run_num)
    # evaluate_model_accuracy(qlearning_dataset_train_final_no_weighting, test_name="Train_No_Weighting", is_debug=True, alpha=alpha, gamma=gamma, run_num=run_num)


# In[71]:

# In[45]:
def final_model_run_parallel(run, train_flag_set, test_flag_set, MIMIC_zscores, all_hyperparameters):
	total_needed_runs:List[int] = [i for i in range(0, len(train_flag_set))]
	thread_needed_runs:List[int] = np.array_split(total_needed_runs, 24)[run]
	for i in thread_needed_runs:
		train_flag = train_flag_set[i]
		test_flag = test_flag_set[i]
        # Cluster to construct modified dataset
		qlearning_dataset_train_final, qlearning_dataset_test_final, transition_mat, physician_policy, total_actions, state_count = aggregate_training_clustering(train_flag, test_flag, MIMIC_zscores)  
		for j in range(0, len(all_hyperparameters)):
			gamma = all_hyperparameters[j][0]
			alpha = all_hyperparameters[j][1]
			# Perform Q-Learning to build finalized dataset with rewards
			qlearning_dataset_train_final_weighted, qlearning_dataset_train_final_no_weighting, qlearning_dataset_test_final_weighted, qlearning_dataset_test_final_no_weighting = aggregate_Q_learning_model_construction(qlearning_dataset_train_final, 
                                                    qlearning_dataset_test_final,
                                                    gamma=gamma, alpha=alpha, state_count=state_count, 
                                                    total_actions=total_actions, physician_policy=physician_policy)
			# Evaluate the functions
			aggregate_Q_learning_evaluation(qlearning_dataset_train_final_weighted, qlearning_dataset_train_final_no_weighting,
                                            qlearning_dataset_test_final_weighted, qlearning_dataset_test_final_no_weighting,
                                            gamma=gamma, alpha=alpha, run_num=i)

if __name__ == "__main__":
    train_flag_set, test_flag_set, MIMIC_zscores = aggregate_dataset_construct()
    all_hyperparameters = calculate_all_hyperparameter_combos()
	# Run over 24 Cores
    jobs = []
    for run in range(25):
        p = multiprocessing.Process(target=final_model_run_parallel, args=(run, train_flag_set, test_flag_set, MIMIC_zscores, all_hyperparameters))
        jobs.append(p)
        p.start()
    # Wait until all processes are finished
    for job in jobs:
        job.join()
    
