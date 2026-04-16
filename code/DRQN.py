# -*- coding: utf-8 -*-
CUDA_LAUNCH_BLOCKING="1"

import math
import random
import os
import numpy as np
from sklearn.cluster import KMeans
from collections import namedtuple
import sparse
import pickle
from itertools import compress
from scipy.special import softmax
from collections import deque
from scipy.sparse import dok_matrix
from os.path import exists

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils




# use cuda if gpu is available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"

seed=int(os.environ.get("SEED", "0"))
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



Transition = namedtuple('Transition',
                        ('pre_session_state','state', 'action', 'next_state', 'reward','page','next_page','did'))


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


def collate_fn(data):
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data, data_length


class DRQN(nn.Module):

    def __init__(self,  f1, h1,f2,h2,h_duel, outputs):
        super(DRQN, self).__init__()
        self.rnn1 = nn.LSTM(f1, h1, 1, batch_first=True)
        self.rnn2 = nn.LSTM(f2, h2, 1, batch_first=True)


        self.fc_adv = nn.Sequential(
            nn.Linear(h2, h_duel),
            nn.ReLU(),
            nn.Linear(h_duel, outputs)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(h2, h_duel),
            nn.ReLU(),
            nn.Linear(h_duel, 1)
        )


    def forward(self, x1, x2):
        output1, (h1n, c1n) = self.rnn1(x1)
        output2, _ = self.rnn2(x2, (h1n, c1n))

        #x2 is the padded data
        out_pad, out_len = rnn_utils.pad_packed_sequence(output2, batch_first=True)
        outcome = out_pad[np.arange(0, x1.shape[0]), out_len - 1, :]

        # Dueling Setting
        val = self.fc_val(outcome)
        adv = self.fc_adv(outcome)
        return val + (adv - adv.mean(dim=1, keepdim=True)),outcome





######################################################################
# Training
# --------
#
# Hyperparameters
######################################################################

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5000"))
GAMMA = 1
TARGET_UPDATE = 50
n_actions = 23
n_pages=3
lag_1=8
lag_2=10
feature_1=15
feature_2=15
hidden_1=n_actions * 2
hidden_2=n_actions * 2
hidden_duelling=math.floor(n_actions * 1.5)
num_episodes = int(os.environ.get("NUM_EPISODES", "800"))
memory_sise=9999999


# Q learning structure setting
policy_net = DRQN(feature_1,hidden_1, feature_2,hidden_2, hidden_duelling, n_actions).to(device)
target_net = DRQN(feature_1,hidden_1, feature_2,hidden_2, hidden_duelling,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(memory_sise)
test_memory = ReplayMemory(memory_sise) # For Testing Set


def optimize_model(double=True):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))


    pre_session_state_batch = torch.cat(batch.pre_session_state).view(BATCH_SIZE,lag_1,feature_1)
    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = batch.next_state

    state_batch_processed=collate_fn(state_batch)
    state_batch_packed = rnn_utils.pack_padded_sequence(state_batch_processed[0], state_batch_processed[1], batch_first=True,enforce_sorted=False)

    next_state_batch_processed=collate_fn(next_state_batch)
    next_state_batch_packed = rnn_utils.pack_padded_sequence(next_state_batch_processed[0], next_state_batch_processed[1], batch_first=True,enforce_sorted=False)

    pre_session_state_batch=pre_session_state_batch.to(device)
    state_batch_packed = state_batch_packed.to(device)
    next_state_batch_packed = next_state_batch_packed.to(device)


    state_action_values,intermediate_state = policy_net(pre_session_state_batch,state_batch_packed)
    state_action_values=state_action_values.gather(1, action_batch)


    # Use double DQN to Compute V(s_{t+1}) for all next states.
    if double:
        next_state_actions = (policy_net(pre_session_state_batch,next_state_batch_packed)[0]).max(1)[1]
        next_state_values = (target_net(pre_session_state_batch,next_state_batch_packed)[0]).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1).detach()
    else:
        next_state_values = (target_net(pre_session_state_batch,next_state_batch_packed)[0]).max(1)[0].detach()

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
        memory.push(torch.tensor(pre_session_state, dtype=torch.float32).to(device),
                    torch.tensor(within_session_pre_state, dtype=torch.float32).to(device),
                    torch.tensor([[action]], dtype=torch.int64).to(device),
                    torch.tensor(within_session_state, dtype=torch.float32).to(device),
                    torch.tensor([reward], dtype=torch.int64).to(device),
                    torch.tensor(page, dtype=torch.int64).to(device),
                    torch.tensor(next_page, dtype=torch.int64).to(device),
                    torch.tensor(did, dtype=torch.int64).to(device),
                    )
    with open("testing_feature_batch.pkl", 'rb') as f:
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

        test_memory.push(torch.tensor(pre_session_state, dtype=torch.float32).to(device),
                    torch.tensor(within_session_pre_state, dtype=torch.float32).to(device),
                    torch.tensor([[action]], dtype=torch.int64).to(device),
                    torch.tensor(within_session_state, dtype=torch.float32).to(device),
                    torch.tensor([reward], dtype=torch.int64).to(device),
                    torch.tensor(page, dtype=torch.int64).to(device),
                    torch.tensor(next_page, dtype=torch.int64).to(device),
                    torch.tensor(did, dtype=torch.int64).to(device),
                    )
else:
    # Load simulated data into Memory for model training
    did = 0
    for x in range(50000):
        # Store the transition in memory
        pre_session_state = torch.randn(lag_1, feature_1)
        state_length = random.randrange(lag_2) + 2
        next_state = torch.randn(state_length, feature_2)
        state = next_state[0:(state_length - 1), :]

        action = torch.tensor([[random.randrange(n_actions)]], device=device)
        reward = torch.tensor([np.random.binomial(1, 0.2) * (random.randrange(10) + 1)], device=device)
        page = random.randrange(n_pages)
        next_page = random.randrange(n_pages)

        memory.push(pre_session_state, state, action, next_state, reward, page, next_page, did)

        if np.random.rand() < 0.1:
            did = did + 1
            print('New ID: ' + str(did))


print("Data Loading Complete")



for i_episode in range(num_episodes):

    print("Round: "+str(i_episode))
    # Perform one step of the optimization (on the target network)
    optimize_model()

    # Update the target network, copying all weights and biases in DRQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Model Training Complete')


# Get intermediate Result
def get_intermedia_state(policy_net,memory):

    transitions = memory.memory
    batch = Transition(*zip(*transitions))


    pre_session_state_batch = torch.cat(batch.pre_session_state).view(len(transitions),lag_1,feature_1)
    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = batch.next_state
    page_batch = batch.page
    next_page_batch = batch.next_page

    state_batch_processed=collate_fn(state_batch)
    state_batch_packed = rnn_utils.pack_padded_sequence(state_batch_processed[0], state_batch_processed[1], batch_first=True,enforce_sorted=False)

    next_state_batch_processed=collate_fn(next_state_batch)
    next_state_batch_packed = rnn_utils.pack_padded_sequence(next_state_batch_processed[0], next_state_batch_processed[1], batch_first=True,enforce_sorted=False)

    pre_session_state_batch=pre_session_state_batch.to(device)
    state_batch_packed = state_batch_packed.to(device)
    next_state_batch_packed = next_state_batch_packed.to(device)


    state_action_values, intermediate_state = policy_net(pre_session_state_batch, state_batch_packed)
    _, next_intermediate_state = policy_net(pre_session_state_batch, next_state_batch_packed)

    return state_action_values,intermediate_state,next_intermediate_state,page_batch,next_page_batch


Q_prediction,intermedia_state,next_intermedia_state,page_batch,next_page_batch=get_intermedia_state(policy_net,memory)
if len(test_memory)>0: # collect intermediate results from testing data, if testing data is available
    _, intermedia_state_test, next_intermedia_state_test, page_batch_test, next_page_batch_test = get_intermedia_state(
        policy_net, test_memory)


# Do clustering for interactions associated with a webpage
page_batch=np.array(page_batch)
next_page_batch=np.array(next_page_batch)
kmeans_result=np.zeros((Q_prediction.shape[0],2))
number_cluster = 200 # set the number of cluster based on elbow method
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
    cluster_count = max(1, min(number_cluster, feature_kmeans.shape[0]))
    kmeans = KMeans(n_clusters=cluster_count, random_state=1)

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
        action_candidate=date_summary[i_T][2][0].cpu().numpy()[0]
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
    Pi_e = np.zeros((cluter_number, action_num))+0.0001
    for i_cluster in range(cluter_number):
        date_summary_cluster = list(compress(date_summary, state_cluster == i_cluster))
        Pi_e[1] = Pi_e[1] / sum(Pi_e[1])
        for i_record in date_summary_cluster:
            i_action = i_record[2]
            Pi_e[i_cluster, i_action] = Pi_e[i_cluster, i_action] + 1
        Pi_e[i_cluster] = Pi_e[i_cluster] / sum(Pi_e[i_cluster])


    # Get M based on Alg4 of Mandel. et. al 2016
    M_Prime = np.ones(cluter_number)
    ratio_matrix = Pi_b / Pi_e
    T_coo = T.to_coo()
    for i_H in range(H):
        # Avoid sparse/dense broadcast errors by reweighting each stored
        # transition explicitly with the current next-state multipliers.
        weighted_values = T_coo.data * M_Prime[T_coo.coords[2]]
        state_action_state_matrix = sparse.COO(T_coo.coords, weighted_values, shape=T_coo.shape)
        state_action_max_state_matrix = state_action_state_matrix.max(axis=2).todense()
        M_Prime = np.asarray((ratio_matrix * state_action_max_state_matrix).max(axis=1)).reshape(-1)

    return M_Prime, Pi_b, Pi_e



Episode_Horizon = int(os.environ.get("EPISODE_HORIZON", "4"))
M,Pi_b,Pi_e=cal_m(Episode_Horizon, n_actions, number_cluster*n_pages,memory.memory,kmeans_result,Q_prediction.cpu().detach().numpy())


# Reject Sampling
def reject_sampling(H,date_summary,M,Pi_b,Pi_e,kmeans_result):

    state_cluster = kmeans_result[:, 0]
    pro_batch = deque(maxlen=H) # store current ratio
    rwd_batch= deque(maxlen=H)  # store current reward
    state_batch = deque(maxlen=H) # store current state
    M_batch= deque(maxlen=H)# store current M


    accpeted_rwd_batch = [] # store accepted episode aggregated reward
    accpeted_ratio_batch = [] # store accepted episode aggregated ratio

    all_rwd_batch = [] # store all episode aggregated reward
    all_ratio_batch = [] # store all episode aggregated ratio

    pre_did=''

    for i_record in range(len(date_summary)):
        did = date_summary[i_record][7]
        action = date_summary[i_record][2].cpu().numpy()[0][0]
        reward=date_summary[i_record][4].cpu().numpy()[0]
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

            # Rejection Sampling
            u = np.random.rand()
            if u<=candidate:
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
