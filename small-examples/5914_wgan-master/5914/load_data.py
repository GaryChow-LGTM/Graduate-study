from read_file import read_csvfile
import numpy as np
def load_data(filepath = "./data.csv"):
    data = read_csvfile(filepath)
    data_X1 = np.array(data[:,1:3], dtype = np.float64)  # 取第二列到第三列的数据
    data_X2 = np.array(data[:,3:], dtype = np.int64) # 取第三列之后的数据
    data_Y = np.array(data[:,0], dtype = np.float64).reshape(-1,1)  # 第一列的结束体积作为预测值
    return data_X1,data_X2, data_Y

if __name__ == '__main__':
    data_X1, data_X2, data_Y = load_data(filepath = "./data.csv")
    print(data_X1)
    print(data_X2)
    print(data_Y)