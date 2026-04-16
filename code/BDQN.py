# -*- coding: utf-8 -*-
CUDA_LAUNCH_BLOCKING="1"

import math
import random
import os
import numpy as np
from collections import namedtuple
from sklearn.cluster import KMeans
from os.path import exists
import pickle
import sparse
from collections import deque
from itertools import compress
from scipy.special import softmax
from scipy.sparse import dok_matrix


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


seed=int(os.environ.get("SEED", "0"))
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"
print(torch.__version__)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','page','next_page','did'))


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

    def __init__(self,f1, f2, f3, outputs):
        super(DQN, self).__init__()
        self.hidden1 = nn.Linear(f1,f2)
        self.hidden2 = nn.BatchNorm1d(f2)
        self.hidden3 = nn.Linear(f2, f3)
        self.hidden4 = nn.BatchNorm1d(f3)
        self.hidden5 = nn.Linear(f3,outputs)



    def forward(self, x):
        x=self.hidden1(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x=self.hidden3(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.hidden5(x)

        return x

def collate_fn(data):
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data, data_length

def BayesReg(batch):

    global E_W
    global E_W_
    global Cov_W
    global E_W_target
    global Cov_W_decom
    global phiphiT
    global phiY

    # Discount previous records
    phiphiT *= (1 - alpha)
    phiY *= (1 - alpha)

    # Data preparation into different parts
    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = batch.next_state

    state_batch_packed = torch.cat(state_batch).view(BATCH_SIZE,feature).to(device)
    next_state_batch_packed = torch.cat(next_state_batch).view(BATCH_SIZE,feature).to(device)


    next_intermediate_state = policy_net(next_state_batch_packed)
    next_intermediate_state_target = target_net(next_state_batch_packed)


    # Get intermediate result
    drqn_coutome = policy_net(state_batch_packed).detach()
    next_state_actions = torch.matmul(next_intermediate_state,torch.transpose(E_W_,0,1)).max(1)[1].detach()
    next_state_values = torch.matmul(next_intermediate_state_target,torch.transpose(E_W_target,0,1)).gather(1,next_state_actions.unsqueeze(-1)).squeeze(-1).detach()
    target_Y=(next_state_values * GAMMA) + reward_batch

    for j in range(BATCH_SIZE):
        bat_action = action_batch[j]-1
        phiphiT[int(bat_action)] += torch.matmul(drqn_coutome[j].view(hidden_out,1),drqn_coutome[j].view(1,hidden_out))
        phiY[int(bat_action)] += drqn_coutome[j]*target_Y[j]


    # Bayesian Posterior for each action
    for i in range(n_actions):
        Cov_W[i] = torch.linalg.inv((phiphiT[i]/sigma_n + (1/sigma)*eye))
        E_W[i] = torch.matmul(Cov_W[i],phiY[i])/sigma_n
        #Cov_W[i] = torch.tensor(sigma * inv)

    for ii in range(n_actions):
        # Cov_W_decom will Used FOR Sampling
        Cov_W_decom[ii] = torch.linalg.cholesky(((Cov_W[ii] + torch.transpose(Cov_W[ii],0,1)) / 2))


# sample model W form the posterior.
def sample_W(E_W,U):
    global E_W_
    for i in range(n_actions):
        sam = torch.normal(mean=0, std=1, size=(hidden_out,1)).to(device)
        E_W_[i] = (E_W[i].clone().detach() + torch.matmul(U[i].clone().detach(),sam.clone().detach())[:,0]).to(device)
    return E_W_


def optimize_model(epoch):
    if len(memory) < BATCH_SIZE:
        return

    double = True

    # Claim Global
    global E_W
    global E_W_
    global Cov_W
    global E_W_target
    global Cov_W_decom
    global phiphiT
    global phiY


    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))


    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = batch.next_state

    state_batch_packed = torch.cat(state_batch).view(BATCH_SIZE,feature).to(device)
    next_state_batch_packed = torch.cat(next_state_batch).view(BATCH_SIZE,feature).to(device)



    intermediate_state = policy_net(state_batch_packed)
    next_intermediate_state = policy_net(next_state_batch_packed)
    next_intermediate_state_target = target_net(next_state_batch_packed)

    # Bayesian Update the last Regression Layer, phiphiT, phiY and E_W, E_Cov will update in the global variable
    E_W_ = sample_W(E_W, Cov_W_decom)
    if epoch<10:
        BayesReg(batch)
        E_W_target = E_W
    elif epoch % Bayesian_UPDATE ==0:
        BayesReg(batch)
        E_W_target = E_W


    state_action_values = torch.matmul(intermediate_state,torch.transpose(E_W_,0,1)).gather(1, action_batch)


    # Use double DQN to Compute V(s_{t+1}) for all next states.
    if double:
        next_state_actions = torch.matmul(next_intermediate_state,torch.transpose(E_W_,0,1)).max(1)[1]
        next_state_values = torch.matmul(next_intermediate_state_target,torch.transpose(E_W_target,0,1)).gather(1,next_state_actions.unsqueeze(-1)).squeeze(-1).detach()
    else:
        next_state_values = torch.matmul(next_intermediate_state_target,torch.transpose(E_W_target,0,1)).max(1)[0].detach()

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


######################################################################
# Training
######################################################################

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5000"))
GAMMA = 1
TARGET_UPDATE = 50
Bayesian_UPDATE = 75


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



# Bayesian Inililization
sigma_n=1
sigma=0.001
alpha=0.01


# All Bayesian Regression Parameters will set requires_grad=False, as they will be updated via Beyesian regression, not via backpropagation
eye = torch.zeros(hidden_out,hidden_out,requires_grad=False).to(device)
for i in range(hidden_out):
    eye[i,i] = 1

E_W = torch.normal(mean=0, std =.01, size=(n_actions,hidden_out),requires_grad=False).to(device)
E_W_target = torch.normal(mean=0, std =.01, size=(n_actions,hidden_out),requires_grad=False).to(device)
E_W_ = torch.normal(mean=0, std =.01, size=(n_actions,hidden_out),requires_grad=False).to(device)
Cov_W = torch.normal(mean=0, std = 1, size=(n_actions,hidden_out,hidden_out),requires_grad=False).to(device)+eye
Cov_W_decom = torch.normal(mean=0, std = 1, size=(n_actions,hidden_out,hidden_out),requires_grad=False).to(device)+eye
for i in range(n_actions):
    Cov_W[i] = eye
    Cov_W_decom[i] = torch.linalg.cholesky(((Cov_W[i]+torch.transpose(Cov_W[i],0, 1))/2))

phiphiT = torch.zeros(n_actions,hidden_out,hidden_out,requires_grad=False).to(device)
phiY = torch.zeros(n_actions,hidden_out,requires_grad=False).to(device)




# Q learning structure setting
policy_net = DQN(f1,f2, f3,hidden_out).to(device)
target_net = DQN(f1,f2, f3,hidden_out).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(memory_sise)
test_memory = ReplayMemory(memory_sise)




# Load Data into Memory
if exists('training_feature_batch.pkl') and exists('testing_feature_batch.pkl'):  # Load real-world training and testing data if it is available
    # Load Real Data
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
    # Initialize the environment and state

    print("Round: "+str(i_episode))
    # Perform one step of the optimization (on the target network)
    optimize_model(i_episode)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Model Training Complete')



# Get intermediate Result
def get_intermedia_state(policy_net,memory):

    transitions = memory.memory
    num_sample = len(memory.memory)
    batch = Transition(*zip(*transitions))

    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = batch.next_state
    page_batch = batch.page
    next_page_batch = batch.next_page


    state_batch_packed = torch.cat(state_batch).view(num_sample,feature).to(device)
    next_state_batch_packed = torch.cat(next_state_batch).view(num_sample,feature).to(device)



    # get intermediate_state and next intermediate_state
    intermediate_state=policy_net(state_batch_packed).detach()
    state_action_values=torch.matmul(intermediate_state, torch.transpose(E_W_, 0, 1))
    next_intermediate_state = policy_net(next_state_batch_packed).detach()

    return state_action_values,intermediate_state,next_intermediate_state,page_batch,next_page_batch


Q_prediction,intermedia_state,next_intermedia_state,page_batch,next_page_batch=get_intermedia_state(policy_net,memory)

if len(test_memory)>0: # collect intermediate results from testing data, if testing data is available
    _, intermedia_state_test, next_intermedia_state_test, page_batch_test, next_page_batch_test = get_intermedia_state(
        policy_net, test_memory)

# Do clustering based on Page associate with the focal interaction
page_batch=np.array(page_batch)
next_page_batch=np.array(next_page_batch)
kmeans_result=np.zeros((Q_prediction.shape[0],2))
number_cluster = 200 # set the number of cluster based on elbow method
kmeans = KMeans(n_clusters=number_cluster, random_state=1)
intermedia_state_cpu=intermedia_state.cpu().detach().numpy()
next_intermedia_state_cpu=next_intermedia_state.cpu().detach().numpy()

if len(test_memory) > 0:  # construct data for model testing, if testing data is available
    page_batch_test = np.array(page_batch_test)
    next_page_batch_test = np.array(next_page_batch_test)
    kmeans_result_test = np.zeros((page_batch_test.shape[0], 2))
    intermedia_state_test = intermedia_state_test.cpu().detach().numpy()
    next_intermedia_state_test = next_intermedia_state_test.cpu().detach().numpy()


for i_page in range(n_pages): # the state before or after the transitions on certain webpage
    select_index=page_batch==i_page
    select_next_index=next_page_batch==i_page
    feature_kmeans=np.concatenate((intermedia_state_cpu[select_index],next_intermedia_state_cpu[select_next_index]), axis=0)

    temp_result = kmeans.fit_predict(feature_kmeans).astype(int)
    temp_result=temp_result+i_page*number_cluster
    kmeans_result[select_index,0]=temp_result[0:sum(select_index)]
    kmeans_result[select_next_index, 1] = temp_result[sum(select_index): (sum(select_index)+sum(select_next_index))]

    if len(test_memory)>0: # collect clustering results from testing data, if testing data is available
        select_index_test = page_batch_test == i_page
        select_next_index_test = next_page_batch_test == i_page
        feature_kmeans_test = np.concatenate(
            (intermedia_state_test[select_index_test], next_intermedia_state_test[select_next_index_test]), axis=0)
        temp_result_test = kmeans.predict(feature_kmeans_test).astype(int)
        temp_result_test = temp_result_test + i_page * number_cluster
        kmeans_result_test[select_index_test, 0] = temp_result_test[0:sum(select_index_test)]
        kmeans_result_test[select_next_index_test, 1] = temp_result_test[
                                              sum(select_index_test): (sum(select_index_test) + sum(select_next_index_test))]

kmeans_result=kmeans_result.astype(int)
if len(test_memory)>0:
    kmeans_result_test = kmeans_result_test.astype(int)



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
        select_Q=Q_prediction[state_cluster == i_cluster] # collect all the record with the state i_cluster
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

# Reject Sampleing
def reject_sampling(H,date_summary,M,Pi_b,Pi_e,kmeans_result):

    state_cluster = kmeans_result[:, 0]
    pro_batch = deque(maxlen=H) # store current ratio
    rwd_batch= deque(maxlen=H)  # store current reward
    state_batch = deque(maxlen=H) # store current state
    M_batch= deque(maxlen=H)# store current M


    accpeted_rwd_batch = [] # store accpeted episode aggregated reward
    accpeted_ratio_batch = [] # store accpeted episode aggregated ratio

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

            # Reject samping
            u = np.random.rand()
            if u<candidate:
                accpeted_rwd_batch.append(np.sum(list(rwd_batch)))
                accpeted_ratio_batch.append(candidate)

            # stroe all episode aggregated info
            all_rwd_batch.append(np.sum(list(rwd_batch)))
            all_ratio_batch.append(candidate)
            #temp_all_rwd_batch.extend(list(rwd_batch))

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
