import numpy as np
import sys, os
import gym
import Environments, Params
import OracleQ, Decoding, QLearning
import argparse
import torch
import random


torch.set_default_tensor_type(torch.DoubleTensor)

def parse_environment_params(args):
    ep_dict = {'horizon': args.horizon,
               'dimension': args.dimension,
               'tabular': args.tabular}
    if args.env_param_1 is not None:
        ep_dict['switch'] = float(args.env_param_1)
    if args.env_param_2 is not None:
        if args.env_param_2 == 'None':
            ep_dict['noise'] = None
        else:
            ep_dict['noise'] = float(args.env_param_2)
    return (ep_dict)

def get_env(name, args):
    env = gym.make(name)
    ep_dict = parse_environment_params(args)
    env.seed(args.seed+args.iteration*31)
    env.init(env_config=ep_dict)
    return(env)

def get_alg(name, args, env):
    if name == "oracleq":
        alg_dict = {'horizon': args.horizon,
                    'alpha': args.lr,
                    'conf': args.conf }
        alg = OracleQ.OracleQ(env.action_space.n, params=alg_dict)
    elif name == 'decoding':
        alg_dict = {'horizon': env.horizon,
                    'model_type': args.model_type,
                    'n': args.n,
                    'num_cluster': args.num_cluster}
        alg = Decoding.Decoding(env.observation_space.n, env.action_space.n,params=alg_dict)
    elif name=='qlearning':
        assert args.tabular, "[EXPERIMENT] Must run QLearning in tabular mode"
        alg_dict = {
            'alpha': float(args.lr),
            'epsfrac': float(args.epsfrac),
            'num_episodes': int(args.episodes)}
        alg = QLearning.QLearning(env.action_space.n, params=alg_dict)
    return (alg)

def parse_args():
    parser = argparse.ArgumentParser(description='StateDecoding Experiments')
    parser.add_argument('--seed', type=int, default=367, metavar='N',
                        help='random seed (default: 367)')
    parser.add_argument('--iteration', type=int, default=1,
                        help="Which replicate number")
    parser.add_argument('--env', type=str, default="Lock-v0",
                        help='Environment', choices=["Lock-v0", "Lock-v1"])
    parser.add_argument('--horizon', type=int, default=4,
                        help='Horizon')
    parser.add_argument('--dimension', type=int, default=5,
                        help='Dimension')
    parser.add_argument('--tabular', type=bool, default=False,
                        help='Make environment tabular')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Training Episodes')
    parser.add_argument('--env_param_1', type=str,
                        help='Additional Environment Parameters (Switching prob)', default=None)
    parser.add_argument('--env_param_2', type=str,
                        help='Additional Environment Parameters (Feature noise)', default=None)
    parser.add_argument('--alg', type=str, default='dqn',
                        help='Learning Algorithm', choices=["oracleq", "decoding", "qlearning"])
    parser.add_argument('--model_type', type=str, default='linear',
                        help='What model class for function approximation', choices=['linear', 'nn'])
    parser.add_argument('--lr', type=float,
                        help='Learning Rate for optimization-based algorithms', default=3e-2)
    parser.add_argument('--epsfrac', type=float,
                        help='Exploration fraction for Baseline DQN.', default=0.1)
    parser.add_argument('--conf', type=float,
                        help='Exploration Bonus Parameter for Oracle Q.', default=3e-2)
    parser.add_argument('--n', type=int, default = 200,
                        help="Data collection parameter for decoding algoithm.")
    parser.add_argument('--num_cluster', type=int, default = 3,
                        help="Num of hidden state parameter for decoding algoithm.")
    args = parser.parse_args()
    return(args)

def train(env, alg, args):
    T = args.episodes
    running_reward = 0
    reward_vec = []
    for t in range(1,T+1):
        state = env.reset()
        done = False
        while not done:
            action = alg.select_action(state)
            next_state, reward, done, _ = env.step(action)
            alg.save_transition(state, action, reward, next_state)
            state = next_state
            running_reward += reward
        alg.finish_episode()
        if t % 100 == 0:
            reward_vec.append(running_reward)
        if t % 1000 == 0:
            print("[EXPERIMENT] Episode %d Completed. Average reward: %0.2f" % (t, running_reward/t), flush=True)
    return (reward_vec)

def main(args):
    random.seed(args.seed+args.iteration*29)
    np.random.seed(args.seed+args.iteration*29)

    import torch
    torch.manual_seed(args.seed+args.iteration*37)

    env = get_env(args.env, args)
    alg = get_alg(args.alg, args, env)


    P = Params.Params(vars(args))
    fname = P.get_output_file_name()
    if os.path.isfile(fname):
        print("[EXPERIMENT] Already completed")
        return None

    reward_vec = train(env,alg,args)

    print("[EXPERIMENT] Learning completed")
    f = open(fname,'w')
    f.write(",".join([str(z) for z in reward_vec]))
    f.write("\n")
    f.close()
    print("[EXPERIMENT] Done")
    return None

if __name__=='__main__':
    Args = parse_args()
    print(Args)
    main(Args)
