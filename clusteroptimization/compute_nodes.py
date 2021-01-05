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
import os
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
# Import the MDP toolbox that contains a method for conducting Q-Learning
# Tool can be found here: https://github.com/sawcordwell/pymdptoolbox
# Documentation for the tool can be found here 
# Itertools provides an easy way to perform Cartesian product on multiple sets
import multiprocessing.dummy
from itertools import product as cartesian_prod

os.system("taskset -p 0xfffff %d" % os.getpid())

### Some repetitive type hinting
int_matrix2D = np.array
float_matrix2D = np.array
int_matrix3D = np.array
float_matrix3D = np.array

### 
#  An MDP, or Markov Decision Process is used to model relationships between various states and actions.
#  A state can be thought of in medical solution as a patient's diagnosis based on current vitals and state of being. 
#  An action can be thought of as a change in current diagnosis based on one of those vitals.
#  The inspirations for the bulk of this code came from Komorowksi's AI Clinician which can be found 
#  here: https://github.com/matthieukomorowski/AI_Clinician/blob/master/AIClinician_core_160219.m
###

###
# Begin by establishing some global variables for use in the MDP creation
###
mdp_count:int = 500            # The number of repititions we want/count of MDPs we need to create 
clustering_iter:int = 32       # The number of times clustering will be conducted
cluster_sample:float = 0.25    # Proportion of the data used for clustering
gamma:float = 0.99             # How close we desire clusters to be in similarity (Percentage)
transition_threshold:int = 5   # The cutoff value for the transition matrix
final_policies:int = 1         # The number of policies we would like to end up with
state_count:int = 750          # The number of distinct states
action_count:int = 5           # Number of actions per state (reccommended 2 to 10)
crossval_iter:int = 5          # Number of crossvalidation runs (Default is 80% Train, 20% Test)
# This will be replaced by the loop index at some point (Iterations of all the models)
loop_index:int = 0

###
# Data structures to hold our interim data
###

def calculate_optimal_clusters_parallel(num:int, data_set_sample:pd.DataFrame):
    data_set=data_set_sample
    max_state_count=750
    num_loops_per_iter=10000
    max_num_iter=clustering_iter
    total_needed_runs:List[int] = [i for i in range(533, 751)]
    thread_needed_runs:List[int] = np.array_split(total_needed_runs, 24)[num]
    # Code for this sample was provided largely by Dr. Giabbanelli 
    # This makes use of the new curvature method for calculating optimal clusters
    # For K-Means sampling
    variance_results:List[float] = np.zeros((max_state_count))
    for i in range(len(thread_needed_runs)):
        state_count = thread_needed_runs[i]
        clusters_models = KMeans(n_clusters=state_count, max_iter=num_loops_per_iter, n_init=max_num_iter).fit(data_set)
        cluster_values = clusters_models.cluster_centers_
        closest_clusters:np.ndarray = vq(train_zscores, cluster_values)
        cluster_distances = closest_clusters[1]
        total_variance = 0
        for i in range(0, len(cluster_distances)):
            total_variance = total_variance + cluster_distances[i]
        variance_results[state_count] = total_variance    
        single_run = f'{state_count},{total_variance}'
        with open('all_cluster_runs.csv', 'a') as f:
            print(single_run, file=f)

if __name__ == '__main__':		
    train_zscores = pickle.load(open("train_zscores.txt", "rb"))
    train_flag = pickle.load(open("sample_train.txt", "rb"))
    sample_train_set:pd.DataFrame = train_zscores[train_flag]
    for i in range(25):
        p = multiprocessing.Process(target=calculate_optimal_clusters_parallel, args=(i, sample_train_set))
        p.start()
        p.join()

            
