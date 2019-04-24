### Sensitivity heatmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from argparse import Namespace
import argparse
import Params
import scipy.stats.mstats


FONTSIZE=12
AXISFONT=10

MAX_ITERS = 25
THRESHOLD = 0.25
FAIL = Params.T + 100

def extrapolate_linearly(x, arr):
    out = np.matrix(np.zeros((len(x),)))
    out[0,0:len(arr)] = arr
    increments = np.array([arr[-1] + (i+1)*arr[-1]/len(arr) for i in range(len(x) - len(arr))])
    out[0,len(arr):] = increments
    return (out)

def load_data():
    x = np.arange(100,Params.T+1,100)

    nvals = Params.lock_dl_n
    kvals = range(2,11,1)
    output = np.matrix(np.zeros((len(kvals), len(nvals))))
    
    Params.reset_params()
    for arg_list in Params.SensitivityParameters['Lock-v0']['decoding']:
        P = Params.Params(arg_list)
        collated = None
        for i in range(1,MAX_ITERS+1):
            P.iteration = i
            fname = P.get_output_file_name()
            try:
                f = open(fname)
            except Exception:
                continue
            tmp = np.loadtxt(f,delimiter=',',dtype=float)
            if collated is None:
                collated = np.matrix(tmp)
            else:
                collated = np.vstack((collated,tmp))
        if collated is None:
            continue
        normalized = collated/x
        val = np.percentile(normalized,50,axis=0)[-1]
        output[kvals.index(P.num_cluster),nvals.index(P.n)] = val

    return(output)

if __name__=='__main__':

    mat = load_data()

    plt.rc('font', size=AXISFONT)
    plt.rc('font', family='sans-serif')

    nvals = Params.lock_dl_n
    kvals = range(2,11,1)

    f = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0]*1, mpl.rcParams['figure.figsize'][1]*0.5))
    ax = f.add_subplot(111)
    im = ax.imshow(mat,extent=[25,1025,11.5,1.5],aspect='auto')
    f.colorbar(im)
#     print([a.get_text() for a in ax.get_xticklabels()])
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#     plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
    ax.set_xlabel('Data Collection Parameter',fontsize=FONTSIZE)
    ax.set_ylabel('Clustering Parameter',fontsize=FONTSIZE)
    ax.set_title('Reward at 100k episodes',fontsize=FONTSIZE)
    plt.savefig('./figs/sensitivity.pdf', format='pdf', dpi=100, bbox_inches='tight')
    plt.close(f)
