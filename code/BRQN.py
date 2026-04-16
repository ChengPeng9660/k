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
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


# set the random seed
seed=int(os.environ.get("SEED", "0"))
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# use cuda if gpu is available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"

print(torch.__version__)


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


    def forward(self, x1, x2):
        output1, (h1n, c1n) = self.rnn1(x1)
        output2, _ = self.rnn2(x2, (h1n, c1n))

        #x2 is the padded data
        out_pad, out_len = rnn_utils.pad_packed_sequence(output2, batch_first=True)
        outcome = out_pad[np.arange(0, x1.shape[0]), out_len - 1, :]


        adv = self.fc_adv(outcome)
        return adv

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


    # Get intermediate results
    drqn_coutome = policy_net(pre_session_state_batch,state_batch_packed).detach()
    next_state_actions = torch.matmul(policy_net(pre_session_state_batch, next_state_batch_packed),torch.transpose(E_W_,0,1)).max(1)[1].detach()
    next_state_values = torch.matmul(target_net(pre_session_state_batch, next_state_batch_packed),torch.transpose(E_W_target,0,1)).gather(1,next_state_actions.unsqueeze(-1)).squeeze(-1).detach()
    target_Y=(next_state_values * GAMMA) + reward_batch

    # Get intermediate variables
    for j in range(BATCH_SIZE):
        bat_action = action_batch[j]-1
        phiphiT[int(bat_action)] += torch.matmul(drqn_coutome[j].view(hidden_out,1),drqn_coutome[j].view(1,hidden_out))
        phiY[int(bat_action)] += drqn_coutome[j]*target_Y[j]


    # Bayesian Posterior update for each action
    for i in range(n_actions):
        Cov_W[i] = torch.linalg.inv((phiphiT[i]/sigma_n + (1/sigma)*eye))
        E_W[i] = torch.matmul(Cov_W[i],phiY[i])/sigma_n

    for ii in range(n_actions):
        # Cov_W_decom will Used for Sampling
        Cov_W_decom[ii] = torch.linalg.cholesky(((Cov_W[ii] + torch.transpose(Cov_W[ii],0,1)) / 2))


# sample parameter W from the posterior.
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

    # Claim global variables
    global E_W
    global E_W_
    global Cov_W
    global E_W_target
    global Cov_W_decom
    global phiphiT
    global phiY


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

    # Bayesian Update the last Regression Layer, phiphiT, phiY and E_W, E_Cov will update in the global variable
    E_W_ = sample_W(E_W, Cov_W_decom)
    if epoch<10:
        BayesReg(batch)
        E_W_target = E_W
    elif epoch % Bayesian_UPDATE ==0:
        BayesReg(batch)
        E_W_target = E_W



    state_action_values = torch.matmul(policy_net(pre_session_state_batch,state_batch_packed),torch.transpose(E_W_,0,1)).gather(1, action_batch)


    # Use Double DQN to Compute V(s_{t+1}) for all next states.
    if double:
        next_state_actions = torch.matmul(policy_net(pre_session_state_batch, next_state_batch_packed),torch.transpose(E_W_,0,1)).max(1)[1]
        next_state_values = torch.matmul(target_net(pre_session_state_batch, next_state_batch_packed),torch.transpose(E_W_target,0,1)).gather(1,next_state_actions.unsqueeze(-1)).squeeze(-1).detach()
    else:
        next_state_values = torch.matmul(target_net(pre_session_state_batch,next_state_batch_packed),torch.transpose(E_W_target,0,1)).max(1)[0].detach()

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



def select_action(state):
    return policy_net(state).max(1)[1].view(1, 1)

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
feature_1=15
feature_2=15
n_pages=3
hidden_1=n_actions * 2
hidden_2=n_actions * 2
hidden_duelling=math.floor(n_actions * 1.8)
hidden_out=math.floor(n_actions * 1.5)
num_episodes = int(os.environ.get("NUM_EPISODES", "800"))
memory_sise=9999999


# Bayesian Parameter Initialization
sigma_n=1
sigma=0.001
alpha=0.01


# All Bayesian Regression Parameters will set requires_grad=False, as they will be updated via Bayesian regression, not via backpropagation
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
policy_net = DRQN(feature_1,hidden_1, feature_2,hidden_2, hidden_duelling, hidden_out).to(device)
target_net = DRQN(feature_1,hidden_1, feature_2,hidden_2, hidden_duelling,hidden_out).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(memory_sise) # For training set
test_memory = ReplayMemory(memory_sise) # For testing set


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
    optimize_model(i_episode)

    # Update the target network, copying all weights and biases from policy BRQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Model Training Complete')




# Get intermediate Result
def get_intermedia_state(policy_net,memory):

    transitions = memory.memory
    batch = Transition(*zip(*transitions))

    # pre_session_state_batch need to change to accommodate varying length
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

    # get intermediate_state and next intermediate_state
    intermediate_state=policy_net(pre_session_state_batch, state_batch_packed)
    state_action_values=torch.matmul(intermediate_state, torch.transpose(E_W_, 0, 1))
    next_intermediate_state = policy_net(pre_session_state_batch, next_state_batch_packed).detach()

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

for i_page in range(n_pages):
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
    Pi_e = np.zeros((cluter_number, action_num))+0.00001
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



def PI(mu, std, best): # probability of improvement acquisition function
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs

def EI(mu,std,best): # expected improvement acquisition function
    with np.errstate(divide='warn'):
        if std>0:
            imp = mu - best - 0.01
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            return ei
    return 0

def TS(mu,std): # thompson sampling acquisition function
    return np.random.normal(mu, std, 1)[0]


def update_model(policy_net,test_memory,E_W,Cov_W,phiphiT,phiY):
    transitions = test_memory.memory
    batch = Transition(*zip(*transitions))
    number_additional_sample = len(transitions)

    pre_session_state_batch = torch.cat(batch.pre_session_state).view(number_additional_sample, lag_1, feature_1)
    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = batch.next_state

    state_batch_processed = collate_fn(state_batch)
    state_batch_packed = rnn_utils.pack_padded_sequence(state_batch_processed[0], state_batch_processed[1],batch_first=True, enforce_sorted=False)

    next_state_batch_processed = collate_fn(next_state_batch)
    next_state_batch_packed = rnn_utils.pack_padded_sequence(next_state_batch_processed[0],next_state_batch_processed[1], batch_first=True, enforce_sorted=False)

    pre_session_state_batch = pre_session_state_batch.to(device)
    state_batch_packed = state_batch_packed.to(device)
    next_state_batch_packed = next_state_batch_packed.to(device)


    # get intermediate_state
    intermediate_state=policy_net(pre_session_state_batch, state_batch_packed)


    #Set the prior distribution for three acquisition function based on the known info
    PI_E_W = E_W.clone().detach()
    PI_COV_W = Cov_W.clone().detach()
    PI_phiphiT=phiphiT.clone().detach()
    PI_phiY=phiY.clone().detach()

    EI_E_W = E_W.clone().detach()
    EI_COV_W = Cov_W.clone().detach()
    EI_phiphiT=phiphiT.clone().detach()
    EI_phiY=phiY.clone().detach()

    TS_E_W = E_W.clone().detach()
    TS_COV_W = Cov_W.clone().detach()
    TS_phiphiT=phiphiT.clone().detach()
    TS_phiY=phiY.clone().detach()

    # evaluate each incoming sample
    print("Start Collecting New Samples to update the Model.")
    for i_sample in range(number_additional_sample):
        PI_Q_Mean = torch.matmul(intermediate_state[i_sample,:], torch.transpose(PI_E_W, 0, 1))
        PI_Q_STD = torch.sqrt(torch.matmul((torch.matmul(intermediate_state[i_sample, :], PI_COV_W)),intermediate_state[i_sample, :]))
        PI_Qrrent_max=max(PI_Q_Mean+2.5*PI_Q_STD) # Here we use mean plus 2.5 unit of standard deviation to present the current maximal Q-value under a state. This can be replaced with the maximal value from real-world empirical data

        EI_Q_Mean = torch.matmul(intermediate_state[i_sample,:], torch.transpose(EI_E_W, 0, 1))
        EI_Q_STD = torch.sqrt(torch.matmul((torch.matmul(intermediate_state[i_sample, :], EI_COV_W)),intermediate_state[i_sample, :]))
        EI_Qrrent_max=max(EI_Q_Mean+2.5*EI_Q_STD) # Here we use mean plus 2.5 unit of standard deviation to present the current maximal Q-value under a state. This can be replaced with the maximal value from real-world empirical data


        TS_Q_Mean = torch.matmul(intermediate_state[i_sample,:], torch.transpose(TS_E_W, 0, 1))
        TS_Q_STD = torch.sqrt(torch.matmul((torch.matmul(intermediate_state[i_sample, :], TS_COV_W)),intermediate_state[i_sample, :]))


        surrogate_value=torch.zeros(n_actions, 3)
        for i_action in range(n_actions):# loop over all actions
            # feed mean and std into three acquisition functions
            surrogate_value[i_action, 0] = PI(PI_Q_Mean[i_action].item(),PI_Q_STD[i_action].item(),PI_Qrrent_max.item())
            surrogate_value[i_action, 1] = EI(EI_Q_Mean[i_action].item(),EI_Q_STD[i_action].item(),EI_Qrrent_max.item())
            surrogate_value[i_action, 2] = TS(TS_Q_Mean[i_action].item(), TS_Q_STD[i_action].item())

        # Get recommended actions
        recommended_action=torch.argmax(surrogate_value, dim=0)

        # If the recommended action matched with the observed action, update the parameter distribution
        if action_batch[i_sample] == recommended_action[0]:
            #print("Action Matched with PI")
            # Update PI related parameter
            next_state_actions = torch.argmax(
                torch.matmul(policy_net(pre_session_state_batch, next_state_batch_packed)[i_sample, :],
                             torch.transpose(PI_E_W, 0, 1)))
            next_state_values = torch.matmul(target_net(pre_session_state_batch, next_state_batch_packed)[i_sample, :],
                                             torch.transpose(PI_E_W, 0, 1)).gather(0, next_state_actions).squeeze(
                -1).detach()
            target_Y = (next_state_values * GAMMA) + reward_batch[i_sample]
            PI_phiphiT[int(action_batch[i_sample])] += torch.matmul(intermediate_state[i_sample].view(hidden_out, 1),
                                                                    intermediate_state[i_sample].view(1, hidden_out))
            PI_phiY[int(action_batch[i_sample])] += intermediate_state[i_sample] * target_Y
            # Bayesian Posterior for the action
            PI_COV_W[int(action_batch[i_sample])] = torch.linalg.inv(
                (PI_phiphiT[int(action_batch[i_sample])] / sigma_n + (1 / sigma) * eye))
            PI_E_W[int(action_batch[i_sample])] = torch.matmul(PI_COV_W[int(action_batch[i_sample])],
                                                               PI_phiY[int(action_batch[i_sample])]) / sigma_n

        if action_batch[i_sample] == recommended_action[1]:
            # Update EI related parameter
            #print("Action Matched with EI")
            next_state_actions = torch.argmax(
                torch.matmul(policy_net(pre_session_state_batch, next_state_batch_packed)[i_sample, :],
                             torch.transpose(EI_E_W, 0, 1)))
            next_state_values = torch.matmul(target_net(pre_session_state_batch, next_state_batch_packed)[i_sample, :],
                                             torch.transpose(EI_E_W, 0, 1)).gather(0, next_state_actions).squeeze(
                -1).detach()
            target_Y = (next_state_values * GAMMA) + reward_batch[i_sample]
            EI_phiphiT[int(action_batch[i_sample])] += torch.matmul(intermediate_state[i_sample].view(hidden_out, 1),
                                                                    intermediate_state[i_sample].view(1, hidden_out))
            EI_phiY[int(action_batch[i_sample])] += intermediate_state[i_sample] * target_Y
            # Bayesian Posterior for the action
            EI_COV_W[int(action_batch[i_sample])] = torch.linalg.inv(
                (EI_phiphiT[int(action_batch[i_sample])] / sigma_n + (1 / sigma) * eye))
            EI_E_W[int(action_batch[i_sample])] = torch.matmul(EI_COV_W[int(action_batch[i_sample])],
                                                               EI_phiY[int(action_batch[i_sample])]) / sigma_n

        if action_batch[i_sample] == recommended_action[2]:
            # Update TS related parameter
            #print("Action Matched with TS")
            next_state_actions = torch.argmax(
                torch.matmul(policy_net(pre_session_state_batch, next_state_batch_packed)[i_sample, :],
                             torch.transpose(TS_E_W, 0, 1)))
            next_state_values = torch.matmul(target_net(pre_session_state_batch, next_state_batch_packed)[i_sample, :],
                                             torch.transpose(TS_E_W, 0, 1)).gather(0, next_state_actions).squeeze(
                -1).detach()
            target_Y = (next_state_values * GAMMA) + reward_batch[i_sample]
            TS_phiphiT[int(action_batch[i_sample])] += torch.matmul(intermediate_state[i_sample].view(hidden_out, 1),
                                                                    intermediate_state[i_sample].view(1, hidden_out))
            TS_phiY[int(action_batch[i_sample])] += intermediate_state[i_sample] * target_Y
            # Bayesian Posterior for the action
            TS_COV_W[int(action_batch[i_sample])] = torch.linalg.inv(
                (TS_phiphiT[int(action_batch[i_sample])] / sigma_n + (1 / sigma) * eye))
            TS_E_W[int(action_batch[i_sample])] = torch.matmul(TS_COV_W[int(action_batch[i_sample])],
                                                               TS_phiY[int(action_batch[i_sample])]) / sigma_n


    Q_Distributuon_B_A = torch.zeros(number_additional_sample*3, 2, 4)

    print("Start Q value distribution comparison")
    # Compare Q value distributions before and after update
    for i_sample in range(number_additional_sample):
        Before_Update_Mean = torch.squeeze(torch.matmul(intermediate_state[i_sample,:], torch.transpose(E_W, 0, 1)))
        Before_Update_STD = torch.sqrt(torch.matmul((torch.matmul(intermediate_state[i_sample, :], Cov_W)),intermediate_state[i_sample, :]))
        Before_Update_top3_Mean, Before_Update_top3_index = torch.topk(Before_Update_Mean,3)
        Before_Update_top3_STD = Before_Update_STD[Before_Update_top3_index]
        Q_Distributuon_B_A[i_sample*3:((i_sample+1)*3),:,0]=torch.cat((Before_Update_top3_Mean.view(3,1),Before_Update_top3_STD.view(3,1)),dim=1)

        After_Update_Mean_PI=torch.squeeze(torch.matmul(intermediate_state[i_sample,:], torch.transpose(PI_E_W, 0, 1)))
        After_Update_STD_PI = torch.sqrt(torch.matmul((torch.matmul(intermediate_state[i_sample, :], PI_COV_W)),intermediate_state[i_sample, :]))
        After_Update_top3_Mean_PI, After_Update_top3_index_PI = torch.topk(After_Update_Mean_PI,3)
        After_Update_top3_STD_PI = After_Update_STD_PI[After_Update_top3_index_PI]
        Q_Distributuon_B_A[i_sample*3:((i_sample+1)*3), :, 1] = torch.cat((After_Update_top3_Mean_PI.view(3,1),After_Update_top3_STD_PI.view(3,1)),dim=1)

        After_Update_Mean_EI=torch.squeeze(torch.matmul(intermediate_state[i_sample,:], torch.transpose(EI_E_W, 0, 1)))
        After_Update_STD_EI = torch.sqrt(torch.matmul((torch.matmul(intermediate_state[i_sample, :],  EI_COV_W)),intermediate_state[i_sample, :]))
        After_Update_top3_Mean_EI, After_Update_top3_index_EI = torch.topk(After_Update_Mean_EI,3)
        After_Update_top3_STD_EI = After_Update_STD_EI[After_Update_top3_index_EI]
        Q_Distributuon_B_A[i_sample*3:((i_sample+1)*3), :, 2] = torch.cat((After_Update_top3_Mean_EI.view(3,1),After_Update_top3_STD_EI.view(3,1)),dim=1)


        After_Update_Mean_TS=torch.squeeze(torch.matmul(intermediate_state[i_sample,:], torch.transpose(TS_E_W, 0, 1)))
        After_Update_STD_TS = torch.sqrt(torch.matmul((torch.matmul(intermediate_state[i_sample, :], TS_COV_W)),intermediate_state[i_sample, :]))
        After_Update_top3_Mean_TS, After_Update_top3_index_TS = torch.topk(After_Update_Mean_TS,3)
        After_Update_top3_STD_TS = After_Update_STD_TS[After_Update_top3_index_TS]
        Q_Distributuon_B_A[i_sample*3:((i_sample+1)*3), :, 3] = torch.cat((After_Update_top3_Mean_TS.view(3,1),After_Update_top3_STD_TS.view(3,1)),dim=1)


    Top1_Mean_Before=torch.mean(Q_Distributuon_B_A[np.arange(0, number_additional_sample*3, 3),0,0])
    Top1_STD_Before = torch.mean(Q_Distributuon_B_A[np.arange(0, number_additional_sample * 3, 3), 1, 0])
    Top2_Mean_Before=torch.mean(Q_Distributuon_B_A[np.arange(1, number_additional_sample*3, 3),0,0])
    Top2_STD_Before = torch.mean(Q_Distributuon_B_A[np.arange(1, number_additional_sample * 3, 3), 1, 0])
    Top3_Mean_Before=torch.mean(Q_Distributuon_B_A[np.arange(2, number_additional_sample*3, 3),0,0])
    Top3_STD_Before = torch.mean(Q_Distributuon_B_A[np.arange(2, number_additional_sample * 3, 3), 1, 0])
    print(" Top 1 Mean (Before Update): "+ str(Top1_Mean_Before.item()))
    print(" Top 1 STD (Before Update): " + str(Top1_STD_Before.item()))
    print(" Top 2 Mean (Before Update): "+ str(Top2_Mean_Before.item()))
    print(" Top 2 STD (Before Update): " + str(Top2_STD_Before.item()))
    print(" Top 3 Mean (Before Update): "+ str(Top3_Mean_Before.item()))
    print(" Top 3 STD (Before Update): " + str(Top3_STD_Before.item()))

    Top1_Mean_After_PI = torch.mean(Q_Distributuon_B_A[np.arange(0, number_additional_sample*3, 3),0,1])
    Top1_STD_After_PI = torch.mean(Q_Distributuon_B_A[np.arange(0, number_additional_sample * 3, 3), 1, 1])
    Top2_Mean_After_PI = torch.mean(Q_Distributuon_B_A[np.arange(1, number_additional_sample*3, 3),0,1])
    Top2_STD_After_PI = torch.mean(Q_Distributuon_B_A[np.arange(1, number_additional_sample * 3, 3), 1, 1])
    Top3_Mean_After_PI = torch.mean(Q_Distributuon_B_A[np.arange(2, number_additional_sample*3, 3),0,1])
    Top3_STD_After_PI = torch.mean(Q_Distributuon_B_A[np.arange(2, number_additional_sample * 3, 3), 1, 1])
    print(" Top 1 Mean (After Update via PI): " + str(Top1_Mean_After_PI.item()))
    print(" Top 1 STD (After Update via PI): " + str(Top1_STD_After_PI.item()))
    print(" Top 2 Mean (After Update via PI): " + str(Top2_Mean_After_PI.item()))
    print(" Top 2 STD (After Update via PI): " + str(Top2_STD_After_PI.item()))
    print(" Top 3 Mean (After Update via PI): " + str(Top3_Mean_After_PI.item()))
    print(" Top 3 STD (After Update via PI): " + str(Top3_STD_After_PI.item()))


    Top1_Mean_After_EI = torch.mean(Q_Distributuon_B_A[np.arange(0, number_additional_sample*3, 3),0,2])
    Top1_STD_After_EI = torch.mean(Q_Distributuon_B_A[np.arange(0, number_additional_sample * 3, 3), 1, 2])
    Top2_Mean_After_EI = torch.mean(Q_Distributuon_B_A[np.arange(1, number_additional_sample*3, 3),0,2])
    Top2_STD_After_EI = torch.mean(Q_Distributuon_B_A[np.arange(1, number_additional_sample * 3, 3), 1, 2])
    Top3_Mean_After_EI = torch.mean(Q_Distributuon_B_A[np.arange(2, number_additional_sample*3, 3),0,2])
    Top3_STD_After_EI = torch.mean(Q_Distributuon_B_A[np.arange(2, number_additional_sample * 3, 3), 1, 2])
    print(" Top 1 Mean (After Update via EI): " + str(Top1_Mean_After_EI.item()))
    print(" Top 1 STD (After Update via EI): " + str(Top1_STD_After_EI.item()))
    print(" Top 2 Mean (After Update via EI): " + str(Top2_Mean_After_EI.item()))
    print(" Top 2 STD (After Update via EI): " + str(Top2_STD_After_EI.item()))
    print(" Top 3 Mean (After Update via EI): " + str(Top3_Mean_After_EI.item()))
    print(" Top 3 STD (After Update via EI): " + str(Top3_STD_After_EI.item()))


    Top1_Mean_After_TS = torch.mean(Q_Distributuon_B_A[np.arange(0, number_additional_sample*3, 3),0,3])
    Top1_STD_After_TS = torch.mean(Q_Distributuon_B_A[np.arange(0, number_additional_sample * 3, 3), 1, 3])
    Top2_Mean_After_TS = torch.mean(Q_Distributuon_B_A[np.arange(1, number_additional_sample*3, 3),0,3])
    Top2_STD_After_TS = torch.mean(Q_Distributuon_B_A[np.arange(1, number_additional_sample * 3, 3), 1, 3])
    Top3_Mean_After_TS = torch.mean(Q_Distributuon_B_A[np.arange(2, number_additional_sample*3, 3),0,3])
    Top3_STD_After_TS = torch.mean(Q_Distributuon_B_A[np.arange(2, number_additional_sample * 3, 3), 1, 3])
    print(" Top 1 Mean (After Update via TS): " + str(Top1_Mean_After_TS.item()))
    print(" Top 1 STD (After Update via TS): " + str(Top1_STD_After_TS.item()))
    print(" Top 2 Mean (After Update via TS): " + str(Top2_Mean_After_TS.item()))
    print(" Top 2 STD (After Update via TS): " + str(Top2_STD_After_TS.item()))
    print(" Top 3 Mean (After Update via TS): " + str(Top3_Mean_After_TS.item()))
    print(" Top 3 STD (After Update via TS): " + str(Top3_STD_After_TS.item()))

# Compare the top-3 Q values for each state before and after the Bayesian update
if len(test_memory)>0:
    update_model(policy_net,test_memory,E_W,Cov_W,phiphiT,phiY)
