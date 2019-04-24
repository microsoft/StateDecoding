import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim

class NNPredictor(object):
    class NNModel(nn.Module):
        def __init__(self, input_dim, num_hidden, output_dim):
            super(NNPredictor.NNModel,self).__init__()
            self.network = nn.Sequential(nn.Linear(input_dim,num_hidden, bias=True),
                                         nn.Sigmoid(),
                                         nn.Linear(num_hidden, output_dim, bias=False))
        def forward(self,x):
            return (self.network(x))
    
    def __init__(self, input_dim, num_hidden, output_dim):
        self.model = NNPredictor.NNModel(input_dim, num_hidden, output_dim)
        self.criterion = nn.MSELoss()

    def fit(self, X,y):
        optimizer = optim.Adagrad(self.model.parameters(), lr = 0.1)
        prev_loss = 0.0
        for i in range(5000):
            total_loss = 0.0
            for a in range(y.shape[1]):
                optimizer.zero_grad()
                preds = self.model(torch.tensor(X))
                loss = self.criterion(preds[:,a], torch.tensor(y[:,a]))
                total_loss += loss
                loss.backward()
                optimizer.step()
            if np.mod(i,100) == 0:
                x = total_loss.detach().numpy()
                if np.round(x,3) == np.round(prev_loss,3):
                    break
                prev_loss = x

    def predict(self, x):
        return self.model(torch.tensor(x)).detach().numpy()

class LinearPredictor(object):
    def __init__(self):
        pass

    def fit(self, X,y):
        X = np.matrix(X)
        y = np.matrix(y)
        self.coef_ = np.linalg.pinv(X.T*X)*X.T*y

    def predict(self,x):
        out = self.coef_.T*np.matrix(x).T
        return(np.array(out.T))

class Decoding(object):
    def get_name(self):
        return "decoding"

    def __init__(self,dimension,num_actions,params={}):

        self.N_exp = 1000
        if 'n' in params.keys():
            self.N_exp = int(params['n'])
        self.N_id = self.N_exp
        self.N_p = self.N_exp


        self.dimension = dimension
        self.num_actions = num_actions

        # possible model types: 'linear', 'nn'

        self.model_type = 'linear'
        if 'model_type' in params.keys():
            self.model_type = str(params['model_type'])
        self.models = {}
        if self.model_type == 'linear':
            print('Decoding algorithm with linear regression.')
        elif self.model_type == 'nn':
            print('Decoding algorithm with neural network model.')

        # number of states per level 
        self.num_cluster = 2
        if 'num_cluster' in params.keys():
            self.num_cluster = params['num_cluster']

        self.horizon = 8
        if 'horizon' in params.keys():
            self.horizon=params['horizon']
        self.horizon_decoding=self.horizon

        self.min_sample_per_cluster = 30

        #the followings are learned parameters

        # learned states, a list that maps a level(h) to learned states at level 
        # each state is represented MK-dimensional embedding
        self.states = []

        #estimated transition probability, a dictionary that maps a level (number) to a dictionary
        #which maps a (state-action pair) to vector
        self.trans_probs = {}

        # estimated policies each of which maximizes the probability of getting to a learned hidden state
        # a dictionary that maps a level (h) into the set of policies for level h
        # policies[h][s] returns the policy that maximizes the prob to s
        # a policy maps a level h to state-action maps for level h.
        self.policies = []
        # initialization no need to do anything
        self.policies.append({})

        self.estimated_probs =[]
        self.estimated_probs.append(np.ones(1))

        # optimal policicies for each level
        self.optimal_policies = []
        # rewards per learned state
        self.state_rewards = []
        # max reward per level
        self.max_rewards =[]

        #optimal policy that maximizes the the probability of getting to the state with big reward
        self.optimal_policy = {}

        #the followings are the temporary parameters
        # the policy being used in the current epsidoes
        self.current_policy = {}

        #Our algorithm consists of exploitation and exploration phase,this is an indicator
        self.is_exploit = 0

        #The followings are used for traing policy
        # the current level of doing exploration
        self.current_exploration_level = 1
        # training data
        #collected contexts for training
        #initialization is used for h =1
        self.training_contexts = np.zeros((self.N_exp,self.dimension))
        #collected training hidden states
        self.training_states = []
        #collected training actions
        self.training_actions = []
        #collected reward for all levels
        self.training_rewards = []
        # number of exploration episodes used for the current level
        # ranging from 0 to N_exp-1
        self.current_exp_number = 0

        # in an episode, we record the current level
        self.current_level = 0


    #implementation of environment api
    def select_action(self, x):
        #initial state
        if self.current_level == 0:
            # check if this is the first time
            if len(self.states) == 0:
                # initialize states
                self.states.append([])
                # add beginning context as the beginning state
                self.states[0].append(x)
                
            self.init_goal()
            hidden_state = self.decoding(x,self.current_level)
            action = self.current_policy[self.current_level][hidden_state]
        else:
            # if haven't reached current exploration level, execute current exploration policy
            if  self.current_level < len(self.current_policy):
                hidden_state = self.decoding(x,self.current_level)
                action = self.current_policy[self.current_level][hidden_state]
            # otherwise just choose random action
            else:
                action = np.random.choice(self.num_actions)
        self.current_level +=1
        return (action)

    # Implementation of environment api
    # save transition data only if currently is exploration and we are at the exploration level
    def save_transition(self, state, action, reward, next_state):
        if (not self.is_exploit) and (self.current_level == self.current_exploration_level):
            #collected contexts for training
            self.training_contexts[self.current_exp_number,:] = next_state
            #collected training hidden states
            hidden_state= self.decoding(state,self.current_level-1)
            self.training_states.append(hidden_state)
            #collected training actions
            self.training_actions.append(action)
            self.training_rewards.append(reward)


    # implementation of environment api
    def finish_episode(self):
        self.current_level = 0
        #continue using exploitation policy, do nothing
        if self.is_exploit:
        # haven't finished exploration in this round, contunue to explore
            self.is_exploit = self.is_exploit
        elif self.current_exp_number < self.N_exp-1:
            self.current_exp_number += 1
        #have collected N_exp data, need to update decoding functions and move to the next level exploration
        #Or change to exploitation mode
        else:
            # update parameters    
            self.learn_model()
            self.learn_states()
            self.learn_trans_prob()
            # update optimal policy for each state
            self.design_policy(self.current_exploration_level)
            self.find_optimal_policy()
            # finish expliration, turn to exploitation mode
            if self.current_exploration_level == self.horizon_decoding:
                self.is_exploit = 1
                self.find_exploitation_policy()
            # next stage exploition
            else:
                self.current_exp_number = 0
                self.current_exploration_level += 1
                print("[DECODING] Exploration Level %d" % (self.current_exploration_level))

                # clean up training data
                self.training_contexts = np.zeros((self.N_exp,self.dimension))
                #collected training hidden states
                self.training_states = []
                #collected training actions
                self.training_actions = []
                #collected reward
                self.training_rewards = []


    # set the goal of this round, whether exploring a specific state or exploiting
    def init_goal(self):
        # alredy enters the exploitation phase, do nothing
        if self.is_exploit:
            self.current_policy = self.optimal_policy
        else:
            #current exploration level, randomly sample a hidden state
            temp_exp_policy = {}
            if self.current_exploration_level > 1:
                prev_level_states = self.states[self.current_exploration_level-1]
                exp_target_state = np.random.choice(len(prev_level_states),1)
                temp_exp_policy = self.policies[self.current_exploration_level-1][exp_target_state[0]]
                policy_h = {}
                # randomly select an action
                for s in range(len(prev_level_states)):
                    policy_h[s] = np.random.choice(self.num_actions)
                temp_exp_policy[self.current_exploration_level-1] = policy_h
            # if we are at the first level exploration stage
            else:
                policy_h = {}
                policy_h[0] = np.random.choice(self.num_actions)
                temp_exp_policy[self.current_exploration_level-1] = policy_h
            self.current_policy = temp_exp_policy


    # maps a context x to its corresponding hidden state at level
    def decoding(self,x,level):
        embedding = self.model_prediction(level,x)
        # nearest neighbor
        hidden_state = self.nearest_neighbor(embedding,level)
        return hidden_state

    # search an embedding (search_point) with in states at level (state_level), return state index
    def nearest_neighbor(self,search_point,level):
        min_id = 0
        min_dist = np.inf
        for i in range(len(self.states[level])):
            if np.linalg.norm(self.states[level][i]-search_point,2) < min_dist:
                min_id = i 
                min_dist = np.linalg.norm(self.states[level][i]-search_point,2)
        return min_id


    # learn decoding model
    def learn_model(self):
        n_last_state = len(self.states[self.current_exploration_level-1])

        if self.model_type == 'linear':
            regr = LinearPredictor()
        elif self.model_type == 'nn':
            regr = NNPredictor(self.training_contexts.shape[1], self.num_cluster, n_last_state*self.num_actions)
        
        # construct labels
        y = np.zeros((self.N_exp,n_last_state*self.num_actions))
        for i in range(self.N_exp):
            hidden_state = self.training_states[i]
            action = self.training_actions[i]
            label = np.zeros((1,n_last_state*self.num_actions))
            label[0,hidden_state*self.num_actions+action]= 1
            y[i,:] =  label

        regr.fit(self.training_contexts, y)

        self.models[self.current_exploration_level] = regr
        

    def model_prediction(self,level,context):
        if level == 0:
            return context
        else:
            return self.models[level].predict([context]).ravel()
            

    # clustering to get representation
    def learn_states(self):
        clustering_contexts = self.training_contexts[0:self.N_id,:]
        clustering_embeddings= []
        for i in range(self.N_id):
            clustering_embeddings.append(self.model_prediction(self.current_exploration_level,clustering_contexts[i,:]))
        clustering_embeddings_matrix = np.asarray(clustering_embeddings)

        # cross-validation
        cur_num_cluster = self.num_cluster
        for i in range(self.num_cluster):
            cur_num_cluster = self.num_cluster - i
            kmeans_result = KMeans(n_clusters=cur_num_cluster, random_state=0).fit(clustering_embeddings_matrix)
            # count number of samples per cluster
            n_sample_counts = np.zeros((cur_num_cluster,1))
            for j in range(len(kmeans_result.labels_)):
                cluster_label = kmeans_result.labels_[j]
                n_sample_counts[cluster_label] += 1
            need_to_dec_num_cluster = 0
            for j in range(len(n_sample_counts)):
                if n_sample_counts[j] < self.min_sample_per_cluster:
                    need_to_dec_num_cluster = 1

            if need_to_dec_num_cluster == 0:
                break

        print("[DECODING] Found %d Clusters" % (cur_num_cluster))
        cur_states = kmeans_result.cluster_centers_.tolist()
        self.states.append(cur_states)


    #estimate transition probability
    def learn_trans_prob(self):
        trans_prob_contexts = self.training_contexts[0:self.N_p]
        # init
        trans_prob = np.zeros((len(self.states[self.current_exploration_level-1]),self.num_actions,len(self.states[self.current_exploration_level])))
        trans_prob_normalizing = np.zeros((len(self.states[self.current_exploration_level-1]),self.num_actions))
        for i in range(self.N_p):
            pre_state = self.training_states[i]
            action = self.training_actions[i]
            state = self.decoding(trans_prob_contexts[i],self.current_exploration_level)
            trans_prob[pre_state,action,state] = trans_prob[pre_state,action,state] + 1
            trans_prob_normalizing[pre_state,action] = trans_prob_normalizing[pre_state,action] + 1
        for i in range(len(self.states[self.current_exploration_level-1])):
            for j in range (self.num_actions):
                trans_prob[i,j,:] = trans_prob[i,j,:] / trans_prob_normalizing[i,j]
        self.trans_probs[self.current_exploration_level-1] = trans_prob
        

    #design optimal policy for the level for each learned state by dynamical programming
    def design_policy(self,level):
        # policy collection
        policies_h = {}
        estimated_probs_h = {}
        # each policy
        for target_s in range(len(self.states[level])):
            # the policy that maximize the probability to target_s
            # policy{h} is a dictionary (policy) which maps a index of the state to an action
            policy_target_s = {}
            # at each level h, estimated_probs{h} returns a dictionary
            # this dictionary maps a state to the probability of getting to the states in current level 
            estimated_probs_target_s = {}
            # init to 0
            for h in range(level+1):
                estimated_probs_target_s[h] = np.zeros((len(self.states[h]),1))
            # initialize the last level, only the target state is set to 1
            estimated_probs_target_s[level][target_s] = 1 
            # use dynamic programming to compute the policy
            for h in range(level):
                # policy at the current level
                policy_h_target_s = {}
                h_reverse = level - h-1
                estimated_prob_h_taget_s = np.zeros((len(self.states[h_reverse]),1))
                trans_prob_h = self.trans_probs[h_reverse]
                for state in range(len(self.states[h_reverse])):
                    best_action = -1
                    max_prob = -1
                    for action in range(self.num_actions):
                        reaching_prob = np.dot(trans_prob_h[state,action,:],estimated_probs_target_s[h_reverse+1])
                        if reaching_prob > max_prob:
                            max_prob = reaching_prob
                            best_action = action
                    estimated_prob_h_taget_s[state] = max_prob
                    policy_h_target_s[state] = best_action
                policy_target_s[h_reverse] = policy_h_target_s
                estimated_probs_target_s[h_reverse] = estimated_prob_h_taget_s
            policies_h[target_s] = policy_target_s
            estimated_probs_h[target_s] = estimated_probs_target_s
        self.policies.append(policies_h)
        self.estimated_probs.append(estimated_probs_h)


    # this function is only used after exploration and use collected reward to find which hidden states gives the most reward
    # then the optimal policy is set to maximize the probability to get to this hidden state
    def find_optimal_policy(self):
        # compute average reward for each state
        avg_rewards= np.zeros((len(self.states[self.current_exploration_level]),1))
        visited_count = np.zeros((len(self.states[self.current_exploration_level]),1))
        for i in range(self.N_exp):
            x = self.training_contexts[i]
            state = self.decoding(x,self.current_exploration_level) 
            r = self.training_rewards[i]
            avg_rewards[state] = avg_rewards[state] + r
            visited_count[state] += 1

        avg_rewards = avg_rewards/visited_count
        self.state_rewards.append(avg_rewards)
        # find max
        best_state = np.argmax(avg_rewards)
        cur_level_max_reward = np.max(avg_rewards)
        self.optimal_policies.append(self.policies[self.current_exploration_level][best_state])
        self.max_rewards.append(cur_level_max_reward)


    def find_exploitation_policy(self):
        print("[DECODING] Model-based exploitation", flush=True)
        policy={}
        # Q function for each level
        # Each one is a state-action pair to real map
        Q_functions = {}
        V_functions = {}
        
        V_functions[self.horizon_decoding] = self.state_rewards[self.horizon_decoding-1]

        for h in range(self.horizon_decoding):
            h_reverse = self.horizon_decoding - h - 1
            policy_h = {}
            Q_function_h = np.zeros((len(self.states[h_reverse]), self.num_actions))
            V_function_h = np.zeros((len(self.states[h_reverse]),1))
            # maps a state 
            policy_h = {}
            
            for i in range(len(self.states[h_reverse])):
                for j in range(self.num_actions):
                    Q_function_h[i,j] = self.trans_probs[h_reverse][i,j,:].dot(V_functions[h_reverse+1])
                max_reward = np.max(Q_function_h[i,:])
                best_action = np.argmax(Q_function_h[i,:])

                V_function_h[i] = self.state_rewards[h_reverse-1][i] + max_reward
                policy_h[i] = best_action
            Q_functions[h_reverse] = Q_function_h
            V_functions[h_reverse] = V_function_h
            policy[h_reverse] = policy_h
        self.optimal_policy = policy


    def state_to_str(self,x):
        return("".join([str(z) for z in x]))
