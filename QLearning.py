import numpy as np
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Schedule(object):
    def __init__(self, n, start, stop):
        self.n = n
        self.start = start
        self.stop = stop

    def get_value(self, i):
        val = self.start - i*(self.start - self.stop)/self.n
        if val < self.stop:
            return self.stop
        return val

class QLearning(object):
    def get_name(self):
        return "qlearning"

    def __init__(self,actions,params={}):
        self.Qs = {}

        self.alpha = 0.1
        if 'alpha' in params.keys():
            self.alpha=params['alpha']

        self.exploration_fraction = 0.1
        if 'epsfrac' in params.keys():
            self.exploration_fraction = params['epsfrac']

        self.total_episodes = int(params['num_episodes'])

        self.EPS_END = 0.01
        self.exploration = Schedule(int(self.exploration_fraction*self.total_episodes),
                                    1.0,
                                    self.EPS_END)

        self.num_episodes = 0
        self.num_actions = actions
        self.traj = []

        print("[QLEARNING] Initialized with parameters: alpha: %s, epsfrac: %s" % (str(self.alpha), str(self.exploration_fraction)), flush=True)
        
    def select_action(self, x):
        s = self.state_to_str(x)
        if s not in self.Qs.keys():
            self.Qs[s] = [0 for a in range(self.num_actions)]
        eps_threshold = self.exploration.get_value(self.num_episodes)
        sample = random.random()
        if sample > eps_threshold:
            Qvals = self.Qs[s]
            action = np.random.choice(np.flatnonzero(Qvals == np.max(Qvals)))
        else:
            action = random.randrange(self.num_actions)
        return (action)

    def get_value(self,x):
        s = self.state_to_str(x)
        if s not in self.Qs.keys():
            ## This should only happen at the end of the episode so put 0 here. 
            self.Qs[s] = [0 for a in range(self.num_actions)]
        Qvals = self.Qs[s]
        return(np.max(Qvals))

    def save_transition(self, state, action, reward, next_state):
        self.traj.append(Transition(state, action, next_state, reward))

    def finish_episode(self):
        self.num_episodes += 1
        for transition in self.traj:
            x = transition.state
            a = transition.action
            r = transition.reward
            xp = transition.next_state
            s = self.state_to_str(x)
            V = 0.0 if xp is None else self.get_value(xp)
            Qvals = self.Qs[s]
            Qvals[a] = (1-self.alpha)*Qvals[a] + self.alpha*(r + V)
        self.traj = []

    def state_to_str(self,x):
        return("".join([str(z) for z in x]))
