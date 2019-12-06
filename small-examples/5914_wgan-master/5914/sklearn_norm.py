from sklearn import preprocessing
import numpy as np
from load_data import load_data





if __name__ == '__main__':
    data_X1, data_X2, data_Y = load_data(filepath = "./data.csv")
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(data_X1)
    print(X_minMax)
