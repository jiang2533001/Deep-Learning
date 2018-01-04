import pandas as pd
import numpy as npi
import matplotlib.pyplot as plt
import sklearn.preprocessing import StandardScaler

def read_data(filename, range_1, range_2):
    data_set = pd.read_csv(filename)
    needed_set = data_set[:, range_1:range_2].values
    
    return needed_set
    
def encode_categorical_data(list):
    c_dict = {}
    num = 1
    new_list = []

    for x in list:
        if c_dict.has_key(x) == False:
            c_dict[x] = num
            num += 1 
        new_list.append(c_dict[x])
    
    return new_list
    

def normalize_data(matrix):
    sc = StandardScaler()
    matrix = sc.fit_transform(matrix)

    return matrix

def train_test_split(matrix):

