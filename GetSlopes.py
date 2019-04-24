import pickle
import numpy as np


def get_slope(y):
    x = np.array([5,10,15,20,30,40,50])

    X = np.matrix(np.zeros((7,2)))
    X[:,1] = 1
    X[:,0] = np.matrix(np.log10(x)).T
    
    Y = np.matrix(np.log10(y)).T
    v = np.linalg.pinv(X.T*X)*X.T*Y
    return (v[0,0])

Data = pickle.load(open('./pkls/Lock-v0_oracleq_linear_0.0_None.pkl', 'rb'))
v = get_slope(np.array(Data[0]['solve']))
print("OracleQ Deterministic Slope: %0.3f" % (v))

Data = pickle.load(open('./pkls/Lock-v0_oracleq_linear_0.1_None.pkl', 'rb'))
v = get_slope(np.array(Data[0]['solve']))
print("OracleQ Stochastic Slope: %0.3f" % (v))

Data = pickle.load(open('./pkls/Lock-v0_decoding_linear_0.0_None.pkl', 'rb'))
v = get_slope(np.array(Data[0]['solve']))
print("Decoding Deterministic Slope: %0.3f" % (v))

Data = pickle.load(open('./pkls/Lock-v0_decoding_linear_0.1_None.pkl', 'rb'))
v = get_slope(np.array(Data[0]['solve']))
print("Decoding Stochastic Slope: %0.3f" % (v))

