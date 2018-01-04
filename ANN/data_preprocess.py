import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess(filename):
    data_set = pd.read_csv(filename)
    x = datat_set.iloc[:, 1:13].values
    print encode_categorical_date(x[:, 1])


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
        
