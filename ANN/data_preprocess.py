import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.preprocessing import StandardScaler

def read_data(filename, unneeded, encode):
    data_set = pd.read_csv(filename)
    for column in unneeded:
        del data_set[column]
    
    for column in encode:
        data_set[column]= encode_categorical_data(data_set[column])

    final_set = data_set.values
    return final_set
    
def encode_categorical_data(data):
    column = data.values
    c_dict = {}
    num = 1
    new_list = []

    for x in column:
        if c_dict.has_key(x) == False:
            c_dict[x] = num
            num += 1 
        new_list.append(c_dict[x])
    
    return new_list
    
def normalize_data(matrix):
    sc = StandardScaler()   
    matrix = sc.fit_transform(matrix)

    return matrix

def train_test_split(matrix, ratio):
    total = len(matrix)
    split = int(ratio * total)
    train_set = matrix[0:split, :]
    test_set = matrix[split:, :]

    return train_set[:, 0:-1], train_set[:,-1], test_set[:, 0:-1], test_set[:, -1]
