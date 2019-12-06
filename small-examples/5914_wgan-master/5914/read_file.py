import csv
import numpy as np
def read_csvfile(filepath = "./data.csv"):
    # with open(filepath, "r") as csvfile:
    #     reader = csv.reader(csvfile)
    #     data = []
    #     for row in reader:
    #         data.append(row)
    # data = np.array(data, dtype = None)
    
    data = np.genfromtxt(filepath, 
        dtype = np.float64,
        skip_header = 1, # 要在文件开头跳过的行数
        delimiter = ",", # 用于分隔值的字符串。 默认情况下，任何连续的空格都用作分隔符。 
        unpack = False) # 是否转置
    return data

if __name__ == '__main__':
    data = read_csvfile()
    print(data)
    print(data.shape)
    print(data["volume_end"])
