import numpy as np
import gym
from gym.spaces import MultiBinary, Discrete, Box

class LockBernoulli(gym.Env):
    """A (stochastic) combination lock environment.
    
    Can configure the length, dimension, and switching probability via env_config"""

    def __init__(self,env_config={}):
        self.initialized=False

    def init(self,env_config={}):
        self.initialized=True
        self.max_reward=1
        self.horizon=2
        self.vstar=0.5
        if 'horizon' in env_config.keys():
            self.horizon = int(env_config['horizon'])
        self.dimension=0
        if 'dimension' in env_config.keys():
            self.dimension = int(env_config['dimension'])
        self.tabular = False
        if 'tabular' in env_config.keys():
            self.tabular = env_config['tabular']

        self.action_space = Discrete(4)
        self.reward_range = (0.0,1.0)
        self.state_space = MultiBinary((self.horizon+1)*3)

        self.observation_space = Box(low=0.0, high=1.0, shape=(3+self.dimension,),dtype=np.float)
        setattr(self.observation_space, 'n', 3+self.dimension)

        if self.tabular:
            self.observation_space = MultiBinary((self.horizon+1)*3)
        self.switch = 0.0
        if 'switch' in env_config.keys():
            self.switch = float(env_config['switch'])

        self.opt_a = gym.spaces.np_random.randint(low=0, high=self.action_space.n, size=self.horizon)
        self.opt_b = gym.spaces.np_random.randint(low=0, high=self.action_space.n, size=self.horizon)
        print("[LOCK] Initializing Combination Lock Environment")
        print("[LOCK] A sequence: ", end="")
        print([z for z in self.opt_a], end=", ")
        print("Switches: ", end="")
        print([(z+1)%4 for z in self.opt_a])
        print("[LOCK] B sequence: ", end="")
        print([z for z in self.opt_b], end=", ")
        print("Switches: ", end="")
        print([(z+1)%4 for z in self.opt_b])        

    def step(self,action):
        if self.h == self.horizon:
            raise Exception("[LOCK] Exceeded horizon")

        r = 0
        rtmp = gym.spaces.np_random.binomial(1,0.5)
        next_state = None
        ## First check for end of episode
        if self.h == self.horizon-1:
            ## Done with episode, need to compute reward
            if self.state == 0 and action == self.opt_a[self.h]:
                next_state = 0
                r = rtmp
            elif self.state == 0 and action == (self.opt_a[self.h]+1) % 4:
                next_state = 1
                r = rtmp
            elif self.state == 1 and action == self.opt_b[self.h]:
                next_state = 1
                r = rtmp
            elif self.state == 1 and action == (self.opt_b[self.h]+1) % 4:
                next_state = 0
                r = rtmp
            else:
                next_state = 2
            self.h +=1
            self.state = next_state
            obs = self.make_obs(self.state)
            return obs, r, True, {}

        ber = gym.spaces.np_random.binomial(1, self.switch)
        ## Decode current state
        r = 0
        if self.state == 0:
            ## In state A
            if action == self.opt_a[self.h]:
                if ber:
                    next_state = 1
                else:
                    next_state = 0
            elif action == (self.opt_a[self.h]+1) % 4:
                if ber:
                    next_state = 0
                else:
                    next_state = 1
            else:
                next_state = 2
        elif self.state == 1:
            ## In state B
            if action == self.opt_b[self.h]:
                if ber:
                    next_state = 0
                else:
                    next_state = 1
            elif action == (self.opt_b[self.h]+1) % 4:
                if ber:
                    next_state = 1
                else:
                    next_state = 0
            else:
                next_state = 2
        else:
            ## In state C
            next_state = 2
        self.h +=1
        self.state = next_state
        obs = self.make_obs(self.state)
        return obs, 0, False, {}

    def make_obs(self,s):
        if self.tabular:
            return np.array([s,self.h])
        else:
            new_x = np.zeros((self.observation_space.n,))
            new_x[s] = 1
            new_x[3:] = gym.spaces.np_random.binomial(1,0.5,(self.dimension,))
            return new_x

    def trim_observation(self,o,h):
        return (o)

    def reset(self):
        if not self.initialized:
            raise Exception("Environment not initialized")
        self.h=0
        self.state=0
        obs = self.make_obs(self.state)
        return (obs)

    def render(self,mode='human'):
        if self.state == 0:
            print("A%d" % (self.h))
        if self.state == 1:
            print("B%d" % (self.h))
        if self.state == 2:
            print("C%d" % (self.h))
        

    def close(self):
        pass

    def seed(self, seed=None):
        gym.spaces.prng.seed(seed)

if __name__=='__main__':
    env = LockBernoulli()
    env.init(env_config={'horizon':10,'dimension':10,'switch':0.1})
    for t in range(20):
        o = env.reset()
        done = False
        h = 0
        while not done:
            env.render()
            print(env.trim_observation(o,h))
            (o,r,done,blah) = env.step(gym.spaces.np_random.randint(low=0,high=env.action_space.n,size=1))
            h += 1
        print("End of episode: r=%d" % (r))
