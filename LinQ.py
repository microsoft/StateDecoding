import numpy as np
from collections import namedtuple
import random
import scipy.sparse.linalg
import scipy.linalg

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def isPSD(A, tol=1e-8):
  (E3,V) = scipy.sparse.linalg.eigs(A,k=1,which='LR')
  return np.max(E3) > -tol

class LinQ(object):
    def get_name(self):
        "LinUCB-Q"

    def __init__(self,dimension,actions,params={}):
        self.weights = []
        self.covs = []
        self.invs = []
        self.old_covs = []
        self.all_feats = []
        self.feats = []
        self.rewards = []
        self.to_update = False

        ## NOTE: all environments have max_reward 1
        self.max_reward = 1

        self.conf = 0.1
        if 'conf' in params.keys():
            self.conf=params['conf']

        if 'horizon' in params.keys():
            self.horizon=params['horizon']

        if 'horizon' in params.keys():
            self.horizon=params['horizon']

        self.num_actions = actions
        self.dimension = dimension

        for h in range(self.horizon):
            self.weights.append(np.matrix(np.zeros((self.dimension*self.num_actions,1))))
            self.covs.append(np.matrix(np.eye(self.dimension*self.num_actions)))
            self.old_covs.append(np.matrix(np.eye(self.dimension*self.num_actions)))
            self.invs.append(np.linalg.pinv(self.covs[-1])) 
            self.all_feats.append(np.matrix(np.zeros((0,self.dimension*self.num_actions))))
            self.feats.append(np.matrix(np.zeros((self.dimension*self.num_actions,0))))
            self.rewards.append(np.matrix(np.zeros((0,1))))

        self.num_updates = 0

        self.update_weights()
        self.num_updates = 0
        self.traj = []
        self.h = 0

        self.t = 0
        print("[LinUCB-Q] Initialized with parameters: conf: %s" % (str(self.conf)), flush=True)
        
    def update_weights(self):
        self.num_updates += 1
        for h in range(self.horizon-1,-1,-1):
            if h < self.horizon-1:
                v = self.get_all_values(h+1)
                tmp = self.rewards[h] + v
            else:
                tmp = self.rewards[h]
            bvec = self.feats[h]*tmp
            self.invs[h] = np.linalg.pinv(self.covs[h])
            self.weights[h] = self.invs[h]*bvec
            self.old_covs[h] = self.covs[h].copy()
    
    def get_q_values(self,s,h):
        if h == self.horizon:
            return [0 for a in range(self.num_actions)]
        w = self.weights[h]
        l = [min(1.0,np.float(w.T*self.featurize(s,a) + self.get_bonus(s,a,h))) for a in range(self.num_actions)]
        return(l)

    def get_value(self,s,h):
        return np.max(self.get_q_values(s,h))
    
    def get_all_values(self,h):
        t1 = self.all_feats[h]*self.weights[h]
        t2 = self.get_all_bonuses(h).T
        tmp = t1+t2
        tmp = np.reshape(tmp, (int(tmp.shape[0]/self.num_actions), self.num_actions))
        tmp = np.max(tmp,axis=1)
        return(tmp)
        
    def get_all_bonuses(self,h):
        tmp = self.conf*np.sqrt(np.diag(self.all_feats[h]*self.invs[h]*self.all_feats[h].T))
        return np.matrix(tmp)

    def get_bonus(self,s,a,h):
        if h == self.horizon:
            return 0
        vec = self.featurize(s, a)
        return self.conf*np.sqrt(vec.T*self.invs[h]*vec)

    def featurize(self,s,a):
        vec = np.matrix(np.zeros((self.dimension*self.num_actions,1)))
        vec[self.dimension*a:self.dimension*(a+1),0] = np.matrix(s).T
        return(vec)

    def select_action(self,s):
        l = self.get_q_values(s,self.h)
        l += np.random.normal(0, 0.0001, size=(self.num_actions,))
        act = np.argmax(l)
        return np.argmax(l)

    def save_transition(self,s,a,r,st):
        vec = self.featurize(s,a)
        self.feats[self.h] = np.hstack((self.feats[self.h], vec))
        self.rewards[self.h] = np.vstack((self.rewards[self.h], np.matrix(r)))
        self.covs[self.h] += vec*vec.T
        ## self.invs[self.h] = self.invs[self.h] - (self.invs[self.h]*vec*vec.T*self.invs[self.h])/(1+vec.T*self.invs[self.h]*vec)
        for at in range(self.num_actions):
            self.all_feats[self.h] = np.vstack((self.all_feats[self.h], self.featurize(s,at).T))
        self.h += 1

    def finish_episode(self):
        self.t += 1
        self.h = 0
        if self.t % 1 == 0:
            for h in range(self.horizon):
                if isPSD(self.covs[h] - (1+1/self.horizon)*self.old_covs[h]):
                    self.update_weights()
                    break
        self.to_update = False

        if self.t % 100 == 0:
            print('[LinQ] Episode %d, Number of updates %d' % (self.t, self.num_updates))
