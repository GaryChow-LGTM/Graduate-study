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
        skip_header = 1,
        delimiter = ",",
        unpack = False)
    return data

if __name__ == '__main__':
    data = read_csvfile()
    print(data)
    print(data.shape)
    print(data["volume_end"])
