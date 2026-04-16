CUDA_LAUNCH_BLOCKING="1"

import math
import random
import os
from collections import namedtuple
import sparse
from itertools import compress
from scipy.special import softmax
from collections import deque
from scipy.sparse import dok_matrix
from sklearn.cluster import KMeans
from os.path import exists
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import scipy as sp
import scipy.stats


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"


# set the random seed
seed=int(os.environ.get("SEED", "0"))
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','page','next_page','did'))

class NormalInverseGamma(object):
    """"
    from https://github.com/tonyduan/conjugate-bayes/blob/master/conjugate_bayes/models.py

    Conjugate prior for a univariate Normal distribution with unknown mean mu,
    variance sigma2. Which is used to model the distribution of Q-values for each state-action pair

    Parameters
    ----------
    m:    prior for N(m, tau2) on the mean mu of the distribution
    tau2: prior for N(m, tau2) on the mean mu of the distribution
    a:    prior for Γ(a, b) on the inverse sigma2 of the distribution
    b:    prior for Γ(a, b) on the inverse sigma2 of the distribution
    """
    def __init__(self, m, tau2, a, b):
        self.__dict__.update({"m": m, "tau2": tau2, "a": a, "b": b})

    def fit(self, x):
        update_dict = {
            "m": (self.m / self.tau2 + len(x) * np.mean(x)) / \
                 (len(x) + 1 / self.tau2),
            "tau2": 1 / (len(x) + 1 / self.tau2),
            "a": self.a + len(x) / 2,
            "b": self.b + np.sum((x - np.mean(x)) ** 2) / 2 + \
                 (np.mean(x) - self.m) ** 2 * len(x) / self.tau2 / \
                 (len(x) + 1 / self.tau2) / 2,
        }
        self.__dict__.update(update_dict)

    def get_marginal_mu(self):
        return sp.stats.t(df=2 * self.a, loc=self.m,
                          scale=(self.tau2 * self.b / self.a) ** 0.5)

    def get_marginal_sigma2(self):
        return sp.stats.invgamma(self.a, scale=self.b)

    def get_posterior_prediction(self):
        return sp.stats.t(df=2 * self.a, loc=self.m,
                          scale=(self.b / self.a * (1 + self.tau2)) ** 0.5)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self,f1, f2, f3, f4, h_duel, outputs):
        super(DQN, self).__init__()
        self.hidden1 = nn.Linear(f1,f2)
        self.hidden2 = nn.BatchNorm1d(f2)
        self.hidden3 = nn.Linear(f2, f3)
        self.hidden4 = nn.BatchNorm1d(f3)
        self.hidden5 = nn.Linear(f3,f4)


        self.fc_adv = nn.Sequential(
            nn.Linear(f4, h_duel),
            nn.ReLU(),
            nn.Linear(h_duel, outputs)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(f4, h_duel),
            nn.ReLU(),
            nn.Linear(h_duel, 1)
        )


    def forward(self, x):
        x=self.hidden1(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x=self.hidden3(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.hidden5(x)

        # Dueling Setting
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True)),x



BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5000"))
GAMMA = 1
TARGET_UPDATE = 50

n_actions = 23
lag_1=8
lag_2=10
feature=15
f1=feature
f2=n_actions * 5
f3=n_actions * 3
f4=n_actions * 2
n_pages=3
hidden_1=n_actions * 2
hidden_2=n_actions * 2
hidden_duelling=math.floor(n_actions * 1.8)
hidden_out=math.floor(n_actions * 1.5)
num_episodes = int(os.environ.get("NUM_EPISODES", "800"))
memory_sise=9999999


policy_net = DQN(f1,f2, f3,f4, hidden_duelling, n_actions).to(device)
target_net = DQN(f1,f2, f3,f4, hidden_duelling,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(memory_sise)
test_memory = ReplayMemory(memory_sise)



def optimize_model(double=True):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))


    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = batch.next_state

    state_batch_packed = torch.cat(state_batch).view(BATCH_SIZE,feature).to(device)
    next_state_batch_packed = torch.cat(next_state_batch).view(BATCH_SIZE,feature).to(device)



    state_action_values,intermediate_state = policy_net(state_batch_packed)
    state_action_values=state_action_values.gather(1, action_batch)


    if double:
        next_state_actions = (policy_net(next_state_batch_packed)[0]).max(1)[1]
        next_state_values = (target_net(next_state_batch_packed)[0]).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1).detach()
    else:
        next_state_values = (target_net(next_state_batch_packed)[0]).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    #print("Loss: " + str(loss))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



# Load Data into Memory
if exists('training_feature_batch.pkl') and exists('testing_feature_batch.pkl'):  # Load real-world training and testing data if it is available
    # Load Real-World Data
    with open("training_feature_batch.pkl", 'rb') as f:
        training_batch = pickle.load(f)
    for record in training_batch:
        pre_session_state = record[0]
        within_session_pre_state = record[1]
        within_session_state = record[2]
        action = record[3]
        reward = record[4]
        page = record[5]
        next_page = record[6]
        did=record[7]

        memory.push(torch.tensor(torch.sum(within_session_pre_state,0), dtype=torch.float32).to(device),
                    torch.tensor([[action]], dtype=torch.int64).to(device),
                    torch.tensor(torch.sum(within_session_state,0), dtype=torch.float32).to(device),
                    torch.tensor([reward], dtype=torch.int64).to(device),
                    torch.tensor(page, dtype=torch.int64).to(device),
                    torch.tensor(next_page, dtype=torch.int64).to(device),
                    torch.tensor(did, dtype=torch.int64).to(device))


    with open("testing.pkl", 'rb') as f:
        testing_batch = pickle.load(f)
    for record in testing_batch:
        pre_session_state = record[0]
        within_session_pre_state = record[1]
        within_session_state = record[2]
        action = record[3]
        reward = record[4]
        page = record[5]
        next_page = record[6]
        did=record[7]

        test_memory.push(torch.tensor(torch.sum(within_session_pre_state,0), dtype=torch.float32).to(device),
                    torch.tensor([[action]], dtype=torch.int64).to(device),
                    torch.tensor(torch.sum(within_session_state,0), dtype=torch.float32).to(device),
                    torch.tensor([reward], dtype=torch.int64).to(device),
                    torch.tensor(page, dtype=torch.int64).to(device),
                    torch.tensor(next_page, dtype=torch.int64).to(device),
                    torch.tensor(did, dtype=torch.int64).to(device))
else:
    # Load simulated data into Memory
    did = 0
    for x in range(50000):
        # Store the transition in memory
        state_length = random.randrange(lag_2) + 2
        next_state = torch.randn(state_length, feature)
        state = next_state[0:(state_length - 1), :]
        state = torch.sum(state, 0)
        next_state = torch.sum(next_state, 0)

        action = torch.tensor([[random.randrange(n_actions)]], device=device)
        reward = torch.tensor([np.random.binomial(1, 0.2) * (random.randrange(10) + 1)], device=device)
        page = random.randrange(n_pages)
        next_page = random.randrange(n_pages)

        memory.push(state, action, next_state, reward, page, next_page, did)

        if np.random.rand() < 0.1:
            did = did + 1
            print('New ID: ' + str(did))


print("Data Loading Complete")


for i_episode in range(num_episodes):

    print("Round: "+str(i_episode))
    # Perform one step of the optimization (on the target network)
    optimize_model()

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


print('Model Training Complete')





# Get intermediate Result
def get_intermedia_state(policy_net,memory):

    transitions = memory.memory
    num_sample=len(memory.memory)
    batch = Transition(*zip(*transitions))

    # pre_session_state_batch need to change to the accommodate varying length
    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = batch.next_state
    page_batch = batch.page
    next_page_batch = batch.next_page


    state_batch_packed = torch.cat(state_batch).view(num_sample,feature).to(device)
    next_state_batch_packed = torch.cat(next_state_batch).view(num_sample,feature).to(device)


    # get intermediate_state and next intermediate_state
    state_action_values,intermediate_state=policy_net(state_batch_packed)
    _,next_intermediate_state = policy_net(next_state_batch_packed)

    return action_batch,intermediate_state,next_intermediate_state,page_batch,next_page_batch, reward_batch


action_batch,intermedia_state,next_intermedia_state,page_batch,next_page_batch,reward_batch=get_intermedia_state(policy_net,memory)
if len(test_memory)>0:
    action_batch_test, intermedia_state_test, next_intermedia_state_test, page_batch_test, next_page_batch_test, reward_batch_test = get_intermedia_state(
        policy_net, test_memory)

#Do clustering based on Page associate with the focal interaction
page_batch=np.array(page_batch)
next_page_batch=np.array(next_page_batch)
action_batch=np.array(action_batch.squeeze(-1).cpu())
reward_batch=np.array(reward_batch.squeeze(-1).cpu())
kmeans_result=np.zeros((page_batch.shape[0],2))
number_cluster = 100 # set the number of cluster based on elbow method
kmeans = KMeans(n_clusters=number_cluster, random_state=1)
intermedia_state_cpu=intermedia_state.cpu().detach().numpy()
next_intermedia_state_cpu=next_intermedia_state.cpu().detach().numpy()

if len(test_memory)>0:
    page_batch_test = np.array(page_batch_test)
    next_page_batch_test = np.array(next_page_batch_test)
    action_batch_test = np.array(action_batch_test.squeeze(-1).cpu())
    reward_batch_test = np.array(reward_batch_test.squeeze(-1).cpu())
    kmeans_result_test = np.zeros((page_batch_test.shape[0], 2))
    intermedia_state_cpu_test = intermedia_state_test.cpu().detach().numpy()
    next_intermedia_state_cpu_test = next_intermedia_state_test.cpu().detach().numpy()


for i_page in range(n_pages): # the state before or after the transitions on certain webpage
    select_index=page_batch==i_page
    select_next_index=next_page_batch==i_page
    feature_kmeans=np.concatenate((intermedia_state_cpu[select_index],next_intermedia_state_cpu[select_next_index]), axis=0)

    temp_result = kmeans.fit_predict(feature_kmeans).astype(int)
    temp_result=temp_result+i_page*number_cluster
    kmeans_result[select_index,0]=temp_result[0:sum(select_index)]
    kmeans_result[select_next_index, 1] = temp_result[sum(select_index): (sum(select_index)+sum(select_next_index))]

    if len(test_memory) > 0:
        select_index_test = page_batch_test == i_page
        select_next_index_test = next_page_batch_test == i_page
        feature_kmeans_test = np.concatenate(
            (intermedia_state_cpu_test[select_index_test], next_intermedia_state_cpu_test[select_next_index_test]), axis=0)

        temp_result_test = kmeans.predict(feature_kmeans_test).astype(int)
        temp_result_test = temp_result_test + i_page * number_cluster
        kmeans_result_test[select_index_test, 0] = temp_result_test[0:sum(select_index_test)]
        kmeans_result_test[select_next_index_test, 1] = temp_result_test[
                                              sum(select_index_test): (sum(select_index_test) + sum(select_next_index_test))]


kmeans_result=kmeans_result.astype(int)
if len(test_memory)>0:
    kmeans_result_test = kmeans_result_test.astype(int)


state_cluster = kmeans_result[:, 0]
next_state_cluster = kmeans_result[:, 1]
Q_prediction = np.zeros((state_cluster.shape[0], n_actions))
# Bayesian update the distribution based on the prior distribution and new observations
for i_cluster in range(number_cluster*n_pages):
    for i_actions in range(n_actions):
        filter1=state_cluster==i_cluster
        filter2=action_batch==i_actions
        filter=np.logical_and(filter1,filter2)
        if sum(filter)>0:
            # there is data associate with the state-action pair
            num_instance=sum(filter)
            data=reward_batch[filter]
            # Set the prior distribution
            model = NormalInverseGamma(m=0, tau2=0.5, a=0.5, b=0.5)
            # Update the model with new observation
            model.fit(data)
            # Sample form the posterior distribution
            pred = model.get_posterior_prediction()
            Q_prediction[filter1,i_actions] = pred.rvs(size=sum(filter1))

for i_actions in range(n_actions): # For those state-action pairs that never get updated, we will use the mean Q-value of that action to fill the matrix
    temp_action=Q_prediction[:,i_actions]
    Q_prediction[Q_prediction[:, i_actions] == 0, i_actions]=Q_prediction[:, i_actions].mean()


Q_prediction=torch.from_numpy(Q_prediction)


def cal_m(H, action_num, cluter_number,date_summary,kmeans_result,Q_prediction):
    state_cluster = kmeans_result[:, 0]
    next_state_cluster = kmeans_result[:, 1]


    # Get T Transition sparse matrix
    T = sparse.DOK((cluter_number, action_num, cluter_number), dtype=np.uint8)
    # Construct state action space
    state_action_space = dok_matrix((cluter_number,action_num), dtype=np.uint8)
    for i_T in range(len(state_cluster)):
        action_candidate=date_summary[i_T][1][0].cpu().numpy()[0]

        T[state_cluster[i_T], action_candidate, next_state_cluster[i_T]] = 1
        state_action_space[state_cluster[i_T], action_candidate] = 1


    # Get Pi_b
    Pi_b = np.zeros((cluter_number, action_num))
    for i_cluster in range(cluter_number):
        select_Q=Q_prediction[state_cluster == i_cluster] # collect all the records with the state i_cluster
        if len(select_Q):
            state_Q=np.average(select_Q, axis=0) # take the average Q-value of all the record
            state_action=state_action_space[i_cluster].toarray()
            temp_line=np.multiply(state_action, state_Q)
            temp_line=np.where(temp_line == 0, -999999999, temp_line)
            Pi_b[i_cluster, :] = softmax(temp_line, axis=1)
        else:
            Pi_b[i_cluster, :]=1/action_num


    # Get Pi_e
    Pi_e = np.zeros((cluter_number, action_num))+0.00001
    for i_cluster in range(cluter_number):
        date_summary_cluster = list(compress(date_summary, state_cluster == i_cluster))
        Pi_e[1] = Pi_e[1] / sum(Pi_e[1])
        for i_record in date_summary_cluster:
            i_action = i_record[1]
            Pi_e[i_cluster, i_action] = Pi_e[i_cluster, i_action] + 1
        Pi_e[i_cluster] = Pi_e[i_cluster] / sum(Pi_e[i_cluster])


    # Get M based on Alg4 of Mandel. et. al 2016
    M_Prime = np.ones(cluter_number)
    ratio_matrix = Pi_b / Pi_e
    for i_H in range(H):
        state_action_state_matrix=np.multiply(T, M_Prime)
        state_action_max_state_matrix=state_action_state_matrix.to_coo().max(axis=2)
        M_Prime=(ratio_matrix*state_action_max_state_matrix.todense()).max(axis=1)

    return M_Prime, Pi_b, Pi_e



Episode_Horizon = int(os.environ.get("EPISODE_HORIZON", "4"))
M,Pi_b,Pi_e=cal_m(Episode_Horizon, n_actions, number_cluster*n_pages,memory.memory,kmeans_result,Q_prediction.cpu().detach().numpy())


# Rejection Sampling
def reject_sampling(H,date_summary,M,Pi_b,Pi_e,kmeans_result):

    state_cluster = kmeans_result[:, 0]
    pro_batch = deque(maxlen=H) # store current ratio
    rwd_batch= deque(maxlen=H)  # store current reward
    state_batch = deque(maxlen=H) # store current state
    M_batch= deque(maxlen=H)# store current M


    accpeted_rwd_batch = [] # store accepted episode aggregated reward
    accpeted_ratio_batch = [] # store accepted episode aggregated ratio

    #temp_all_rwd_batch = []
    all_rwd_batch = [] # store all episode aggregated reward
    all_ratio_batch = [] # store all episode aggregated ratio

    pre_did=''

    for i_record in range(len(date_summary)):
        did = date_summary[i_record][6]
        action = date_summary[i_record][1].cpu().numpy()[0][0]
        reward=date_summary[i_record][3].cpu().numpy()[0]
        current_state = state_cluster[i_record]
        current_pi_b = Pi_b[current_state,action]
        current_pi_e = Pi_e[current_state,action]
        current_M=M[current_state]

        if did!=pre_did: # Ensure the consecutive records come from the same user
            pro_batch.clear()
            rwd_batch.clear()
            state_batch.clear()
            M_batch.clear()


        if current_pi_b>0.00001 and current_pi_e>0.00001:
            current_ratio=current_pi_b/current_pi_e
            rwd_batch.append(reward)
            pro_batch.append(current_ratio)
            state_batch.append(current_state)
            M_batch.append(current_M)

        if len(pro_batch)==H:
            candidate_ratio=np.prod(list(pro_batch))
            candidate_M=M_batch[0]
            candidate=candidate_ratio/candidate_M

            # Reject sampling
            u = np.random.rand()
            if u<candidate:
                accpeted_rwd_batch.append(np.sum(list(rwd_batch)))
                accpeted_ratio_batch.append(candidate)

            # store all episode aggregated info
            all_rwd_batch.append(np.sum(list(rwd_batch)))
            all_ratio_batch.append(candidate)

        pre_did=did



    mean_a = np.mean(all_rwd_batch)
    mean_b = np.mean(accpeted_rwd_batch)
    accepted_count = len(accpeted_rwd_batch)
    total_count = len(all_rwd_batch)
    acceptance_rate = accepted_count / total_count if total_count else float("nan")
    print('\nmean for all reward:', mean_a)
    print('mean for accepted reward:', mean_b)
    print('accepted episodes:', accepted_count)
    print('total candidate episodes:', total_count)
    print('acceptance rate:', acceptance_rate)



# Do Rejection Sampling
if len(test_memory)>0:
    print("start rejection sampling")
    reject_sampling(Episode_Horizon, test_memory.memory, M, Pi_b, Pi_e, kmeans_result_test)
