import numpy as np
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class OracleQ(object):
    def get_name(self):
        return "oracleq"

    def __init__(self,actions,params={}):
        self.Qs = {}
        self.Ns = {}

        ## NOTE: all environments have max_reward 1
        self.max_reward = 1

        self.alpha = 0.1
        if 'alpha' in params.keys():
            self.alpha=params['alpha']

        self.conf = 0.1
        if 'conf' in params.keys():
            self.conf=params['conf']

        if 'horizon' in params.keys():
            self.horizon=params['horizon']

        self.num_actions = actions
        self.traj = []

        print("[ORACLEQ] Initialized with parameters: alpha: %s, conf: %s" % (str(self.alpha), str(self.conf)), flush=True)

    def select_action(self, x):
        s = self.state_to_str(x)
        if s not in self.Qs.keys():
            self.Qs[s] = [self.max_reward for a in range(self.num_actions)]
            self.Ns[s] = [0 for a in range(self.num_actions)]
        Qvals = self.Qs[s]
        action = np.random.choice(np.flatnonzero(Qvals == np.max(Qvals)))
        return (action)

    def get_value(self,x):
        s = self.state_to_str(x)
        if s not in self.Qs.keys():
            ## This should only happen at the end of the episode so put 0 here. 
            self.Qs[s] = [0 for a in range(self.num_actions)]
            self.Ns[s] = [0 for a in range(self.num_actions)]
        Qvals = self.Qs[s]
        return(np.max(Qvals))

    def save_transition(self, state, action, reward, next_state):
        self.traj.append(Transition(state, action, next_state, reward))

    def finish_episode(self):
        for transition in self.traj:
            x = transition.state
            a = transition.action
            r = transition.reward
            xp = transition.next_state
            s = self.state_to_str(x)
            self.Ns[s][a] += 1
            V = 0.0 if xp is None else self.get_value(xp)
            Qvals = self.Qs[s]
            Qvals[a] = (1-self.alpha)*Qvals[a] + self.alpha*(r + V + self.conf*np.sqrt(self.horizon/self.Ns[s][a]))
            Qvals[a] = np.minimum(Qvals[a], 1)
        self.traj = []

    def state_to_str(self,x):
        return("".join([str(z) for z in x]))
