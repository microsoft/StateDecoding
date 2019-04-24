import numpy as np
import gym
from gym.spaces import MultiBinary, Discrete, Box
import LockBernoulli

class LockGaussian(LockBernoulli.LockBernoulli):
    """A (stochastic) combination lock environment.
    Here the feature vector is hit with a random rotation and augmented with gaussian noise.

    x = Rs + eps where s is the one-hot encoding of the state.

    Can configure the length, dimension, and switching probability via env_config"""

    def __init__(self,env_config={}):
        super(LockGaussian,self).__init__(env_config=env_config)
        self.initialized=False

    def init(self,env_config={}):
        super(LockGaussian,self).init(env_config=env_config)

        self.noise = None
        if 'noise' in env_config.keys():
            self.noise = env_config['noise']

        self.rotation = np.matrix(np.eye(self.observation_space.n))

    def make_obs(self, s):
        if self.tabular:
            return np.array([s,self.h])
        else:
            gaussian = np.zeros((self.observation_space.n,))
            if self.noise is not None:
                gaussian = gym.spaces.np_random.normal(0,self.noise,[self.observation_space.n])
            gaussian[s] += 1
            x = (self.rotation*np.matrix(gaussian).T).T
            return np.reshape(np.array(x), x.shape[1])

        
if __name__=='__main__':
    env = LockGaussian()
    env.init(env_config={'horizon':2,'dimension': 2, 'tabular': False, 'swap': 0.0, 'noise':0.1})
    for t in range(20):
        o = env.reset()
        done = False
        while not done:
            env.render()
            print(o)
            (o,r,done,blah) = env.step(gym.spaces.np_random.randint(low=0,high=env.action_space.n,size=1))
        print("End of episode: r=%d" % (r))
