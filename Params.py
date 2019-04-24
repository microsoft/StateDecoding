import numpy as np
import itertools

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

class Params(object):
    def __init__(self, dict=None): 
        ### Defaults
        self.iteration=1
        self.env='Lock-v0'
        self.horizon=2
        self.tabular=False
        self.episodes=100000
        self.env_param_1=0.1
        self.env_param_2=None

        self.alg='oracleq'
        self.model_type='linear'
        self.lr=3e-2
        self.epsfrac=0.1
        self.conf=3e-2
        self.n=100
        self.num_cluster=3
        for (k,v) in dict.items():
            setattr(self,k,v)

    def get_param_str(self):
        string = "--iteration %s --env %s --horizon %s --episodes %s --env_param_1 %s --env_param_2 %s --alg %s --model_type %s --lr %s --epsfrac %s --conf %s --n %s --num_cluster %s " % (
            str(self.iteration), self.env, str(self.horizon), str(self.episodes), str(self.env_param_1), 
            str(self.env_param_2), self.alg, self.model_type, str(self.lr), str(self.epsfrac), 
            str(self.conf), str(self.n), str(self.num_cluster))
        if self.tabular:
            string += " --tabular True"
        else:
            string += " --dimension %d" % (self.horizon)
        return (string)

    def __str__(self):
        s = '%s_%s_model=%s_T=%s_H=%s_d=%s_ep1=%s_ep2=%s_lr=%s_epsfrac=%s_conf=%s_n=%s_cluster=%s_iteration=%s' % (
            self.env, self.alg, self.model_type, str(self.episodes), str(self.horizon), 
            '0' if self.tabular else str(self.horizon), str(self.env_param_1),
           str(self.env_param_2), str(self.lr), str(self.epsfrac), str(self.conf), str(self.n), 
           str(self.num_cluster), str(self.iteration))
        return (s)

    def get_output_file_name(self):
        fname = "./data/%s.out" % (self.__str__())
        return(fname)


#### TrainingTimes
T = 100000

LockEpisodes = {
    'oracleq': [100000],
    'decoding': [100000],
    'qlearning': [100000],
    'qlearning_fail': [1000000],
}

#### Horizons
LockHorizons = {
    'oracleq' : [5,10,15,20,30,40,50],
    'decoding': [5,10,15,20,30,40,50],
    'qlearning': [5,10,15,20,30,40,50],
}

#### Noise levels
LockNoises = [None,0.1,0.2,0.3]
### OracleQ Params
oq_conf = np.logspace(-4,0,5)
oq_lr = np.logspace(-4,0,5)

### Decoding Params
lock_dl_n = range(100,1001,100)
lock_dl_num_clusters = [3]

### QLearning Params
ql_lr = np.logspace(-4,0,5)
ql_epsfrac = [0.0001,0.001,0.01,0.1,0.5]


Parameters = {}
SensitivityParameters = {}
def reset_params():
    Parameters['Lock-v0'] = {}
    Parameters['Lock-v0']['oracleq'] = product_dict(env=['Lock-v0'],horizon=LockHorizons['oracleq'],conf=oq_conf, episodes=LockEpisodes['oracleq'], lr=oq_lr, alg=['oracleq'], tabular=[True], env_param_1=[0.0, 0.1])
    Parameters['Lock-v0']['decoding'] = product_dict(env=['Lock-v0'],horizon=LockHorizons['decoding'],n=lock_dl_n, num_clusters=lock_dl_num_clusters, episodes=LockEpisodes['decoding'], alg=['decoding'], model_type=['linear'], env_param_1=[0.0, 0.1])
    Parameters['Lock-v0']['qlearning'] = product_dict(env=['Lock-v0'],horizon=LockHorizons['qlearning'],epsfrac=ql_epsfrac, episodes=LockEpisodes['qlearning'], lr=ql_lr, alg=['qlearning'], tabular=[True], env_param_1=[0.0, 0.1])
    Parameters['Lock-v0']['qlearning_fail'] = product_dict(env=['Lock-v0'],horizon=[15,20],epsfrac=ql_epsfrac, episodes=LockEpisodes['qlearning_fail'], lr=ql_lr, alg=['qlearning'], tabular=[True], env_param_1=[0.0, 0.1])

    Parameters['Lock-v1'] = {}
    Parameters['Lock-v1']['decoding'] = product_dict(env=['Lock-v1'],horizon=LockHorizons['decoding'],n=lock_dl_n, num_clusters=lock_dl_num_clusters, episodes=LockEpisodes['decoding'], alg=['decoding'], model_type=['linear','nn'], env_param_1=[0.0, 0.1], env_param_2=LockNoises)

#     Parameters['Lock-v2'] = {}
#     Parameters['Lock-v2']['decoding'] = product_dict(env=['Lock-v2'],horizon=LockHorizons['decoding'],n=lock_dl_n, num_clusters=lock_dl_num_clusters, episodes=LockEpisodes['decoding'], alg=['decoding'], model_type=['nn','linear'], env_param_1=[0.0, 0.1], env_param_2=LockNoises)

    SensitivityParameters['Lock-v0'] = {}
    SensitivityParameters['Lock-v0']['decoding'] = product_dict(env=['Lock-v0'],horizon=[20],n=lock_dl_n,num_cluster=range(2,11,1),episodes=LockEpisodes['decoding'],alg=['decoding'],model_type=['linear'],env_param_1=[0.1])

    return 

reset_params()
