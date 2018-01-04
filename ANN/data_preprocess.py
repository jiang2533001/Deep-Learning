import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess(filename):
    data_set = pd.read_csv(filename)
    x = data_set.iloc[:, 3:13].values
    y = data_set.iloc[:, 14].values

    x[:, 1] = encode_categorical_data(x[:, 1])
    x[:, 2] = encode_categorical_date(x{:, 2})

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
        
