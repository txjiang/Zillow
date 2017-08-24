import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import minmax_scale, StandardScaler

def read_file(print_head_tail = False):
    row_data = pd.read_csv('properties_2016.csv')
    print (row_data.shape)
    if print_head_tail == True:
        print (row_data.head())
        print (row_data.tail())
    return row_data

def fill_nan(row_data, fill_method):
    prop_land_ID = {31:1,
                    46:2,
                    47:3,
                    246:4,
                    247:5,
                    248:6,
                    260:7,
                    261:8,
                    262:9,
                    263:10,
                    264:11,
                    265:12,
                    266:13,
                    267:14,
                    268:15,
                    269:16,
                    270:17,
                    271:18,
                    273:19,
                    274:20,
                    275:21,
                    276:22,
                    279:23,
                    290:24,
                    291:25}
    row_data = row_data.drop('columpropertyzoningdesc', 1)
    row_data['PropertyLandUseTypeID'].replace(prop_land_ID, inplace = True)
    if fill_method == "mean":
        for col in row_data:
            col.fillna((col.mean()), inplace=True)
    elif fill_method == "normal_distribution":
        for col in row_data:
            col_mean = np.nanmean(col)
            col_std = np.nanstd(col)
            nan_count = np.isnan(col).shape[0]
            fill_normal = np.random.normal(col_mean, col_std, nan_count)
            col[np.isnan(col)] = fill_normal
    else:
        print ("Please define a fill method")
    return row_data

def split_train_test_save(row_data, split_pc = 30, save = True, normalize_metod = "minmax"):
    if normalize_metod == "minmax":
        scaler = minmax_scale()
