import numpy as np
import gym
from gym.envs.registration import register
import LockBernoulli, LockGaussian

register(
    id='Lock-v0',
    entry_point='LockBernoulli:LockBernoulli',
)
register(
    id='Lock-v1',
    entry_point='LockGaussian:LockGaussian',
)
