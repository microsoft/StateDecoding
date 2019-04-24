import numpy as np
import sys
from argparse import Namespace
import argparse
import Params
import scipy.stats.mstats
import Environments
import gym

final_data = {
}

MAX_ITERS = 25

def extrapolate_linearly(x, arr):
    out = np.matrix(np.zeros((len(x),)))
    out[0,0:len(arr)] = arr
    increments = np.array([arr[-1] + (i+1)*arr[-1]/len(arr) for i in range(len(x) - len(arr))])
    out[0,len(arr):] = increments
    return (out)

def get_p_idx(data,p,threshold,fail):
    perc = np.percentile(data,p,axis=0)
    (ix,) = np.where(perc > threshold)
    if len(ix) != 0:
        idx = 100*(ix[0]+1)
    else:
        idx = fail
    return(idx)

def get_f_idx(data,p,fail):
    quantized = np.minimum(data, 1)
    perc = np.mean(quantized,axis=0)
    (ix,iy) = np.where(perc > p/100)
    if len(ix) != 0:
        idx = 100*(iy[0]+1)
    else:
        idx = fail
    return(idx)

def load_data(alg,env,model,ep1,ep2):
    print("Loading data for %s %s %s %s %s" % (alg, env, model, ep1, ep2), flush=True)
    arr = {'solve':[],
           'solve_low':[],
           'solve_high':[],
           'find':[],
           'find_low':[],
           'find_high':[]}
    solve_best = {}
    find_best = {}

    ### Solve is when average reward is 0.5*vstar
    E = gym.make(env)
    E.init(env_config={'dimension':1})
    threshold= 0.5*E.vstar
    T = Params.LockEpisodes[alg][0]
    x = np.arange(100,T+1,100)
    fail = 100*(len(x)+1)
    Params.reset_params()
    solve_median = {}
    solve_high = {}
    solve_low = {}
    find_median = {}
    find_high = {}
    find_low = {}
    hyperparams = {}
    for arg_list in Params.Parameters[env][alg]:
        P = Params.Params(arg_list)
        if str(P.env_param_1) != ep1 or str(P.env_param_2) != ep2 or str(P.model_type) != model:
            continue
        collated = None
        for i in range(1,MAX_ITERS+1):
            P.iteration = i
            fname = P.get_output_file_name()
            try:
                f = open(fname)
            except Exception:
                continue
            tmp = np.loadtxt(f,delimiter=',',dtype=float)
            tmp2 = extrapolate_linearly(x,tmp)
            if collated is None:
                collated = np.matrix(tmp2)
            else:
                collated = np.vstack((collated,tmp2))
        if collated is None:
            continue
        if P.horizon not in hyperparams.keys():
            hyperparams[P.horizon] = []
            solve_median[P.horizon] = []
            solve_high[P.horizon] = []
            solve_low[P.horizon] = []
            find_median[P.horizon] = []
            find_high[P.horizon] = []
            find_low[P.horizon] = []
        hyperparams[P.horizon].append(str(P))
        normalized = collated/x
        solve_median[P.horizon].append(get_p_idx(normalized, 50, threshold, fail))
        solve_low[P.horizon].append(get_p_idx(normalized, 90, threshold, fail))
        solve_high[P.horizon].append(get_p_idx(normalized, 10, threshold, fail))
        find_median[P.horizon].append(get_f_idx(collated, 50, fail))
        find_low[P.horizon].append(get_f_idx(collated, 10, fail))
        find_high[P.horizon].append(get_f_idx(collated, 90, fail))

    ### Now that we have preprocessed, find best parameter for each horizon
    lst = list(hyperparams.keys())
    lst.sort()
    for h in lst:
        idx = None
        min = np.min(solve_high[h])
        if idx is None and min < fail:
            idx = np.argmin(solve_high[h])
        min = np.min(solve_median[h])
        if idx is None and min < fail:
            idx = np.argmin(solve_median[h])
        min = np.min(solve_low[h])
        if idx is None and min < fail:
            idx = np.argmin(solve_low[h])
        if idx is None:
            print("SOLVE: H=%d, Time=Failure" % (h), flush=True)
            arr['solve'].append(fail)
            arr['solve_high'].append(fail)
            arr['solve_low'].append(fail)
            solve_best[h] = None
        else:
            arr['solve'].append(solve_median[h][idx])
            arr['solve_high'].append(solve_high[h][idx])
            arr['solve_low'].append(solve_low[h][idx])
            print("SOLVE: H=%d, Median=%d, Low=%d, High=%d" % (h, arr['solve'][-1], arr['solve_low'][-1], arr['solve_high'][-1]), flush=True)
            solve_best[h] = hyperparams[h][idx]
        idx = None
        min = np.min(find_high[h])
        if idx is None and min < fail:
            idx = np.argmin(find_high[h])
        min = np.min(find_median[h])
        if idx is None and min < fail:
            idx = np.argmin(find_median[h])
        min = np.min(find_low[h])
        if idx is None and min < fail:
            idx = np.argmin(find_low[h])
        if idx is None:
            print("FIND: H=%d, Time=Failure" % (h), flush=True)
            arr['find'].append(fail)
            arr['find_high'].append(fail)
            arr['find_low'].append(fail)
            find_best[h] = None
        else:
            arr['find'].append(find_median[h][idx])
            arr['find_high'].append(find_high[h][idx])
            arr['find_low'].append(find_low[h][idx])
            print("FIND: H=%d, Median=%d, Low=%d, High=%d" % (h, arr['find'][-1], arr['find_low'][-1], arr['find_high'][-1]), flush=True)
            find_best[h] = hyperparams[h][idx]
    arr['horizons'] = lst
    return (arr, find_best, solve_best)

def parse_args():
    parser = argparse.ArgumentParser(description='StateDecoding Postprocessing Script')
    parser.add_argument('--env', type=str, default="Lock-v0",
                        help='Environment', choices=["Lock-v0", "Lock-v1", "Lock-v2"])
    parser.add_argument('--alg', type=str, default="decoding",
                        help='Environment', choices=["decoding", "oracleq", "qlearning"])
    parser.add_argument('--model_type', type=str, default="linear",
                        help='Base Learner', choices=["nn", "linear"])
    parser.add_argument('--env_param_1', type=str, default="0.0",
                        help='Environment', choices=["0.0", "0.1"])
    parser.add_argument('--env_param_2', type=str, default="None",
                        help='Environment parameter', choices=["None", "0.1", "0.2", "0.3", "0.5"])
    args = parser.parse_args()
    return(args)

if __name__=='__main__':
    args = parse_args()
    (arr, find_best, solve_best) = load_data(args.alg, args.env, args.model_type, args.env_param_1, args.env_param_2)
    if args.alg == 'qlearning':
        (fail,find_fail, solve_fail) = load_data('qlearning_fail', args.env, args.model_type, args.env_param_1, args.env_param_2)
        arr['solve'][2] = fail['solve'][0]
        arr['find'][3] = fail['find'][1]
        arr['solve_low'][2] = fail['solve_low'][0]
        arr['find_low'][3] = fail['find_low'][1]
        arr['solve_high'][2] = fail['solve_high'][0]
        arr['find_high'][3] = fail['find_high'][1]
        find_best[20] = find_fail[20]
        solve_best[15] = solve_fail[15]
        print(fail)

    import pickle
    pickle.dump((arr, find_best, solve_best), open("./pkls/%s_%s_%s_%s_%s.pkl" %(args.env, args.alg, args.model_type, args.env_param_1, args.env_param_2), "wb"))

