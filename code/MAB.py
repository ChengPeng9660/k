from sklearn import linear_model
import random
import os
import numpy as np
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


# Set up the random seed
seed=int(os.environ.get("SEED", "0"))
random.seed(seed)
np.random.seed(seed)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','page','next_page','state_action','did'))


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


# Set up the data size
BATCH_SIZE = 5000
GAMMA = 1

n_actions = 23
lag_1=8
lag_2=10
feature=15
n_pages=3
memory_sise=9999999


# Set up the model and the space
memory = ReplayMemory(memory_sise)
test_memory = ReplayMemory(memory_sise)
model = linear_model.BayesianRidge()


# Load Real-World Data
if exists('training_feature_batch.pkl') and exists('testing_feature_batch.pkl'):
    # Load Real-World training Data
    with open("training_feature_batch.pkl", 'rb') as f:
        training_batch = pickle.load(f)
    for feature in training_batch:
        pre_session_state = feature[0]
        within_session_pre_state = feature[1]
        within_session_state = feature[2]
        action = feature[3]
        reward = feature[4]
        page = feature[5]
        next_page = feature[6]
        did=feature[7]
        state = torch.sum(within_session_pre_state, 0).detach().numpy()
        next_state = torch.sum(within_session_state, 0).detach().numpy()
        state_action = np.append(state, action)

        memory.push(state, action, next_state, reward, page, next_page, state_action, did)


    with open("testing.pkl", 'rb') as f: # Load Real-World testing Data
        testing_batch = pickle.load(f)
    for feature in testing_batch:
        pre_session_state = feature[0]
        within_session_pre_state = feature[1]
        within_session_state = feature[2]
        action = feature[3]
        reward = feature[4]
        page = feature[5]
        next_page = feature[6]
        did = feature[7]
        state = torch.sum(within_session_pre_state, 0).detach().numpy()
        next_state = torch.sum(within_session_state, 0).detach().numpy()
        state_action = np.append(state, action)

        test_memory.push(state, action, next_state, reward, page, next_page, state_action, did)
else:
    # Load simulated data into Memory
    did = 0
    for x in range(50000):
        # Store the transition in memory
        state_length = random.randrange(lag_2) + 2
        next_state = torch.randn(state_length, feature)
        state = next_state[0:(state_length - 1), :]
        state = torch.sum(state, 0).detach().numpy()
        next_state = torch.sum(next_state, 0).detach().numpy()

        action = random.randrange(n_actions)
        reward = np.random.binomial(1, 0.2) * (random.randrange(10) + 1)
        page = random.randrange(n_pages)
        next_page = random.randrange(n_pages)
        state_action = np.append(state,action)

        memory.push(state, action, next_state, reward, page, next_page,state_action, did)

        if np.random.rand() < 0.1:
            did = did + 1
            print('New ID: ' + str(did))

print("Data Loading Complete")

transitions = memory.memory
batch = Transition(*zip(*transitions))
state_batch = batch.state
action_batch = batch.action
reward_batch = batch.reward
next_state_batch = batch.next_state
state_action_batch = batch.state_action
page_batch = batch.page
next_page_batch = batch.next_page


X=np.array(state_action_batch)
Y=np.array(reward_batch)


print("Start Model Training")
# Fit the data into the model
model.fit(X, Y)
print('Model Training Complete')


if len(test_memory)>0:
    transitions_test = test_memory.memory
    batch_test = Transition(*zip(*transitions_test))
    state_batch_test = batch_test.state
    action_batch_test = batch_test.action
    reward_batch_test = batch_test.reward
    next_state_batch_test = batch_test.next_state
    state_action_batch_test = batch_test.state_action
    page_batch_test = batch_test.page
    next_page_batch_test = batch_test.next_page

    X_test = np.array(state_action_batch_test)
    intermedia_state_test = np.array(state_batch_test)
    next_intermedia_state_test = np.array(next_state_batch_test)
    page_batch_test = np.array(page_batch_test)
    next_page_batch_test = np.array(next_page_batch_test)
    Q_prediction_test = np.zeros((len(X_test), n_actions))
    kmeans_result_test = np.zeros((len(X_test), 2))

# Predict the reward for different actions under different states
intermedia_state_cpu=np.array(state_batch)
next_intermedia_state_cpu=np.array(next_state_batch)
next_page_batch=np.array(next_page_batch)
page_batch=np.array(page_batch)

Q_prediction=np.zeros((len(X),n_actions))
for i_action in range(n_actions):
    # Get the posterior mean and STD for each Gaussian distribution
    mean_post,std_post = model.predict(np.append(intermedia_state_cpu,np.ones((len(intermedia_state_cpu),1))*i_action,1), return_std=True)
    # sample a Q-value for each state action pair
    Q_prediction[:, i_action] =np.random.normal(mean_post, std_post, len(X))

    if len(test_memory) > 0:
        # Get the posterior mean and STD for each Gaussian distribution
        mean_post_test, std_post_test = model.predict(
            np.append(intermedia_state_test, np.ones((len(intermedia_state_test), 1)) * i_action, 1), return_std=True)
        # sample a Q-value for each state-action pair
        Q_prediction_test[:, i_action] = np.random.normal(mean_post_test, std_post_test, len(X_test))


#Do clustering based on Page associate with the focal interaction

kmeans_result=np.zeros((Q_prediction.shape[0],2))
number_cluster = 200 # set the number of cluster based on elbow method

for i_page in range(n_pages): # the state before or after the transitions with a certain webpage
    select_index=page_batch==i_page
    select_next_index=next_page_batch==i_page
    feature_kmeans=np.concatenate((intermedia_state_cpu[select_index],next_intermedia_state_cpu[select_next_index]), axis=0)
    cluster_count = max(1, min(number_cluster, feature_kmeans.shape[0]))
    kmeans = KMeans(n_clusters=cluster_count, random_state=1)

    temp_result = kmeans.fit_predict(feature_kmeans).astype(int)
    temp_result=temp_result+i_page*number_cluster
    kmeans_result[select_index,0]=temp_result[0:sum(select_index)]
    kmeans_result[select_next_index, 1] = temp_result[sum(select_index): (sum(select_index)+sum(select_next_index))]

    if len(test_memory) > 0:
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

        action_candidate=date_summary[i_T][1]

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



Episode_Horizon = int(os.environ.get("EPISODE_HORIZON", "1"))
M,Pi_b,Pi_e=cal_m(Episode_Horizon, n_actions, number_cluster*n_pages,memory.memory,kmeans_result,Q_prediction)




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
        action = date_summary[i_record][1]
        reward=date_summary[i_record][3]
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

            # Rejection sampling
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
