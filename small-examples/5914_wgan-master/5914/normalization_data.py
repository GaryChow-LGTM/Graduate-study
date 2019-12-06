from load_data import load_data
import numpy as np
def normalization_data(filepath = "./data.csv"):
    data_X1, data_X2, data_Y = load_data(filepath)
    #data Y是肿瘤结束体积
    data_Y/=1000
    #data X1是肿瘤初始体积 Ki67
    data_X1[:,0]/=1000
    data_X1[:,1]/=1
    #data X2是其他各项指标的数据输入
    data_X2 = data_X2.astype(np.float64)
    data_X2[:,0]/=3
    data_X2[:,1]/=3
    data_X2[:,2]/=3
    data_X2[:,3]/=1
    data_X2[:,4] = (data_X2[:,4]-30)/(100-30)
    data_X2[:,5]/=1
    data_X2[:,6]/=4
    data_X2[:,7]/=3
    data_X2[:,8]/=2
    data_X2[:,9]/=3
    data_X2[:,10] = (data_X2[:,10]-40)/(70-40)
    return np.concatenate((data_Y, data_X1, data_X2), axis = 1)

def unnormalization_data(dataset):
    data_X1, data_X2, data_Y = dataset[:,1:3], dataset[:,3:], dataset[:,0].reshape(-1,1)
    #data Y是肿瘤结束体积
    data_Y*=1000
    #data X1是肿瘤初始体积 Ki67
    data_X1[:,0]*=1000
    data_X1[:,1]*=1
    #data X2是其他乱七八槽输入
    data_X2[:,0]*=3
    data_X2[:,1]*=3
    data_X2[:,2]*=3
    data_X2[:,3]*=1
    data_X2[:,4] = data_X2[:,4]*(100-30)+30
    data_X2[:,5]*=1
    data_X2[:,6]*=4
    data_X2[:,7]*=3
    data_X2[:,8]*=2
    data_X2[:,9]*=3
    data_X2[:,10] = data_X2[:,10]*(80-40)+40
    data_X2 = np.around(data_X2).astype(np.int64)
    # print(data_X1, data_X2, data_Y)
    return np.concatenate((data_Y, data_X1, data_X2), axis = 1)
    

if __name__ == '__main__':
    dataset = normalization_data(filepath = "./data.csv")
    print(unnormalization_data(dataset))
    # print(data_X1)
    # print(data_X2)
    # print(data_Y)