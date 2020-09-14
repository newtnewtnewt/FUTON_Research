###
#  FUTON Model MDP + Q-Learning Creation Script
#  A Research Project conducted by Noah Dunn 
###

# Import the standard tools for working with Pandas dataframe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shelve
import typing
from typing import *
# Import the MDP toolbox that contains a method for conducting Q-Learning
# Tool can be found here: https://github.com/sawcordwell/pymdptoolbox
# Documentation for the tool can be found here 
import mdptoolbox

###
# A function to actually perform Q learning
# REPLACE WITH ACTUAL INFORMATION
### 
def offpolicy_Q_learning(ql_train_set: pd.DataFrame, gamma: float, alpha: float, numtraces: int) -> List[List[int]]:
    # We need to save the Q-value for each run 
    sum_Q_values = np.zeros(numtraces)
    # We need to construct the full matrix based on actions and states
    num_actions = len(ql_train_set['chosen_action_index'].unique()) - 1
    num_clusters = len(ql_train_set['closest_cluster_index'].unique())
    # Where the Q-Values are saved at each given run
    # print("Solider boy", num_clusters, num_actions)
    # q_values = np.zeros((num_clusters, num_actions))
    # A perfect Q-value for a given action is the max that can be obtained
    max_avg_Q = 1
    # Modulus for use in processing later
    modulus_val = 100
    # List of the rows that contain the first instance of a patient's data per patient
    first_index_list = ql_train_set[ql_train_set['training_bloc'] == 1].index
    print(first_index_list)
    
    result_matrix = np.zeros((25, 750))
    return result_matrix

###
# parallel_bootql_Creation is a function intended to be used
# with the multithraeding package availabe in Python
# It takes arguments passed in from it's parent function: offpolicy_eval_tdlearning
# unique_training_set_ids: All the patient ids for the training set
# proportion: The proportion of the data to be used for training
# gamma:  A multiplier to be used in the Q-learning
# qlearning_train_dataset_final: The full dataset of value
# physician_policy: A 25 x 750 matrix of the actions a phyiscian chosen given a state (physician_policy[A][S])
# distribution_values: A 750 long array that stores frequency of state appears
###
def parallel_bootql_creation(unique_training_set_ids: pd.DataFrame, proportion: float, gamma: float, qlearning_train_dataset_final: pd.DataFrame, physician_policy: List[List[int]], distribution_values: List[int]) -> List[List[int]]:
    # Grab a random sample of the ids to use for the Q-LEARNING step
    train_len = len(unique_training_set_ids)
    # We are going to randomly mark around (proportion) number of IDs
    # and mark them as 1's to filter
    id_flags = [np.floor(np.random.rand() + proportion) == 1 for i in range(0, train_len)]
    chosen_ids = unique_training_set_ids[id_flags]
    # Choose the rows with training_ids begin at the 1
    ql_train_set_proto = qlearning_train_dataset_final[qlearning_train_dataset_final['training_set_id'].isin(chosen_ids)]
    ql_train_set = pd.concat((ql_train_set_proto['training_bloc'],
                             ql_train_set_proto['closest_cluster_index'], 
                             ql_train_set_proto['chosen_action_index'], 
                             ql_train_set_proto['reward_value']),
                             axis=1,
                             keys=['training_bloc',"closest_cluster_index","chosen_action_index","reward_value"]
                            )
    
            
    # Use the train set to achieve optimal Q-Equation
    offpolicy_Q_result = offpolicy_Q_learning(ql_train_set, gamma, 0.1, 30000)
    
    # Value the phyisican's decision based on Q-Learning probabilistic 
    # distribution and the actual chosen policy at each step
    value_matrix = np.zeros((25, 750))
    for i in range(0, 25):
        for j in range(0, 750):
            value_matrix[i][j] = physician_policy[i][j] * offpolicy_Q_result[i][j]
    
    # The reward of a given state is the sum of all possible actions 
    # That can be taken from that state
    value_sums = np.zeros(750)
    for i in range(0, 25):
        for j in range(0, 750):
            value_sums[i] = value_sums[i] + value_matrix[i, j]
    
    # Return the rewards for each state, weighted by their probability of
    # occuring (gathered from frequency in the actual data)
    return np.nansum(np.multiply(value_sums, distribution_values))/sum(distribution_values)