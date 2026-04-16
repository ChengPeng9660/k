from collections import deque
import numpy as np
import random
import pandas as pd
import pickle
from collections import deque
import copy
import os.path
from os import path
import matplotlib.pyplot as plt
from dateutil import parser
import datetime
from numpy import random



def extract_page(x):
    index=x.find(' - ')
    if index > 0:
        return x[0:(index+1)].strip().lower()
    elif x == 'order':
        return x
    else:
        index = x.find(' ')
        if index > 0:
            return x[0:(index + 1)].strip().lower()
        else:
            return ''

def remove_tz(x):
    return x[0:len(x)-4]


def transform_to_seconds(x):
    return int((x-datetime.datetime(1970, 1, 1, 0, 0, 0)).total_seconds())

def update_page_vector(page_v,record,pre_record):
    if(len(pre_record)>0):
        gap = record['gap']

        pre_page = pre_record['page']
        pre_gap = pre_record['gap']


        if pre_page !='order':
            try:
                page_index=page_list.index(pre_page)
                page_v[page_index] = page_v[page_index] + (gap - pre_gap)
            except ValueError:
                j=1
        return page_v
    else:# new user or new session
        return page_v


def update_product_vector(product_v,record):
    quantity = record['quantity']
    cents = record['totalcents']
    productcode = record['productcode']
    try:
        product_index = product_code_list.index(productcode)
        if product_index >= 0:
            product_v[product_index] = product_v[product_index] + quantity
        product_v[product_length - 1] = product_v[product_length - 1] + cents # the last dimension record the total spending for all the product
    except ValueError:
        print('Error Prodcut Matching')
    return product_v


def get_arm(subject,user_id):
    sub_df=subject[subject['analysis_userid']==user_id]
    treatment = [0] * 10
    for index, row in sub_df.iterrows():
        e_id=row['experimentid']
        arm=row['arm']
        if arm!='control': # all the control arm are control group
            treatment[e_id-160]=1
            print(str(e_id-160)+" treated")
    return treatment

def get_paction_space(click_summary_new):
    temp_space=click_summary_new['treatment'].unique()
    action_space=[]
    for temp in temp_space:
        if temp.startswith( 'e' ): # all the action/action combination start with e
            action_space.append(temp)
    return action_space




def get_treatment(arm_vector,page,time,experiment):
    result=''
    for i in range(len(arm_vector)): # the user might be treated in multiple experiments
        if arm_vector[i]==1:
            exp_start=experiment.loc[i, 'gap_s']
            exp_end = experiment.loc[i, 'gap_e']
            focal_page = experiment.loc[i, 'Page']
            if page==focal_page and time>=exp_start and time<=exp_end: # within the valid experimental date
                result=result+'e'+str(i)
    return result





click_list=['footprints000000000000','footprints000000000001','footprints000000000002','footprints000000000003','footprints000000000004','footprints000000000005','footprints000000000006','footprints000000000007','footprints000000000008','footprints000000000009','footprints000000000010']
# this is list of file name that contains the click-stream data

page_list=['page_1', 'page_2', 'page_3', 'page_4', 'page_5', 'page_6', 'page_7', 'page_8', 'page_9'] # the page_list contains all the pages, where the user activities we are going to collect
product_code_list=['category_1', 'category_2', 'category_3', 'category_4', 'category_5'] # the product_list contains all the products, where the user activities we are going to collect

remaining_title=['session_start','session_end','session_index','userid']
click_remaining_title=['gap','page','treatment','session_index','within_session_index','userid']
session_summary=[]
page_length=len(page_list) # Different page will be considered
product_length=len(product_code_list) # Different product will be considered
click_index=0
session_gap=12*60*60 # the sessions are apart if two consecutive interactions are more than 12 hours


# Pre-session vector construction
session_index=1
pre_user_id=0
pre_page=''
pre_gap=-500
session_begin=-500
pre_row=[]
page_vector=[0] * page_length
product_vector=[0] * product_length

if path.exists("session_summary.pkl"):
    with open("session_summary.pkl", 'rb') as f:
        session_summary = pickle.load(f)
else:
    # iteratively load the click data
    while click_index<len(click_list):
        click_stream = pd.read_csv(click_list[click_index],encoding = "ISO-8859-1") # extract the page for all column and add the column to the dataframe

        click_stream['dt'] = click_stream['dt'].apply(remove_tz)
        click_stream['dt']=pd.to_datetime(click_stream['dt'])
        click_stream['gap'] = click_stream['dt'].apply(transform_to_seconds)
        click_stream['page']=click_stream['action'].apply(extract_page)
        click_stream['page'] = click_stream['page'].replace('product', 'products')
        click_stream.to_pickle(click_list[click_index]+".pkl")

        for index, row in click_stream.iterrows():
            user_id=row['userid']
            page = row['page']
            gap= row['gap'] # in seconds from 1970-01-01
            type=row['type']
            quantity = row['quantity']
            cents=row['totalcents']

            if user_id ==pre_user_id:

                if (gap-pre_gap)<=session_gap: # Same user, same session
                    # update the page_vector for the same user and same session, no need to update page_vector with a new session
                    page_vector=update_page_vector(page_vector, row,pre_row)
                    if page == 'order':
                        product_vector = update_product_vector(product_vector, row)
                else: # Same user, new session
                    print('New Session')
                    session_end=pre_gap
                    session_index = session_index + 1
                    # store the session record
                    temp_list = [session_begin, session_end, session_index, pre_user_id]
                    session_list=page_vector+product_vector+temp_list
                    session_summary.append(session_list)

                    # reset the session tracking info
                    pre_row = []
                    page_vector = [0] * page_length
                    product_vector = [0] * product_length
                    page_vector=update_page_vector(page_vector, row,pre_row)
                    if page == 'order':
                        product_vector = update_product_vector(product_vector, row)
                    session_begin=gap


                pre_page = page
                pre_gap = gap
                pre_row = row
                if page=='order':
                    product_vector=update_product_vector(product_vector, row)
            else: # New user
                print('New User')
                session_end = pre_gap
                # store the previous session record
                temp_list = [session_begin, session_end, session_index, pre_user_id]
                session_list = page_vector + product_vector + temp_list
                session_summary.append(session_list)



                # reset the user tracking info
                session_index = 1
                pre_user_id = user_id
                session_begin = gap
                pre_row = []
                page_vector = [0] * page_length
                product_vector = [0] * product_length
                page_vector = update_page_vector(page_vector, row, pre_row)
                if page == 'order':
                    product_vector = update_product_vector(product_vector, row)
                pre_page = page
                pre_gap = gap
                pre_row = row

        click_index=click_index+1

    with open('session_summary.pkl', 'wb') as f:
        pickle.dump(session_summary, f)




# # within-session vector construction
subject=pd.read_csv('experiment_list.csv',encoding = "ISO-8859-1")
experiment=pd.read_csv('experiment.csv',encoding = "ISO-8859-1")
click_title=['click_gap','page','treatment','session_index','userid']
click_summary=[]


minimal_time=experiment['gap_s'].min()
click_index=0
session_index=1
within_session_index=1
pre_user_id=0
pre_page=''
pre_gap=-500
session_begin=-500
pre_row=[]
page_vector=[0] * page_length
product_vector=[0] * product_length
arm_vector=[0] * 10 # Corresponding to 10 experiments


if path.exists("click_summary.pkl"):
    with open("click_summary.pkl", 'rb') as f:
        click_summary = pickle.load(f)
else:
    # iteratively load the click data
    while click_index<len(click_list):
        click_stream = pd.read_pickle(click_list[click_index] + ".pkl")

        for index, row in click_stream.iterrows():
            user_id=row['userid']
            page = row['page']
            gap=row['gap']
            type=row['type']
            quantity = row['quantity']
            cents=row['totalcents']

            if user_id ==pre_user_id:


                if gap-pre_gap<=session_gap: # Same user, same session
                    # update the page_vector and product_vector

                    if gap>=minimal_time: # start constructing the features since experiment start
                        if (page=='experiment_page_1') or (page=='experiment_page_2'): # the experiments are build on a specific webpage
                            # store the pre-action behavior
                            print('Click: New Action')
                            treatment=get_treatment(arm_vector, page,gap, experiment)
                            if treatment=='':
                                if page=='products':
                                    treatment = 'products_control'
                                elif page=='builder':
                                    treatment = 'builder_control'
                            else:
                                print('New treatment: '+str(treatment))
                            temp_feature_list = [gap-pre_gap, page,treatment, session_index,within_session_index, pre_user_id]
                            feature_list = page_vector + product_vector + temp_feature_list
                            click_summary.append(feature_list)


                            within_session_index = within_session_index + 1
                            page_vector = [0] * page_length
                            product_vector = [0] * product_length
                            page_vector = update_page_vector(page_vector, row, pre_row)
                            if page == 'order':
                                product_vector = update_product_vector(product_vector, row)
                        else:
                            page_vector = update_page_vector(page_vector, row, pre_row)
                            if page == 'order':
                                product_vector = update_product_vector(product_vector, row)
                else: # Same user, new session
                    print('Click: New Session')
                    session_end=pre_gap

                    if gap >= minimal_time: # start constructing the features since experiment start
                        treatment = 'session_trans'
                        session_index = session_index + 1
                        within_session_index = 1

                        temp_feature_list = [gap - pre_gap, page, treatment, session_index, within_session_index,pre_user_id]
                        feature_list = page_vector + product_vector + temp_feature_list
                        click_summary.append(feature_list)

                        within_session_index = within_session_index + 1


                    # reset the session tracking info
                    pre_row = []
                    page_vector = [0] * page_length
                    product_vector = [0] * product_length
                    page_vector=update_page_vector(page_vector, row,pre_row)
                    if page == 'order':
                        product_vector = update_product_vector(product_vector, row)
                    session_begin=gap


                pre_page = page
                pre_gap = gap
                pre_row = row
            else: # New user
                print('Click: New User')
                session_end = pre_gap

                # get the user treatment_info
                arm_vector = get_arm(subject, user_id)
                # reset the user tracking info
                session_index = 1
                within_session_index=1
                pre_user_id = user_id
                session_begin = gap
                pre_row = []
                page_vector = [0] * page_length
                product_vector = [0] * product_length
                page_vector = update_page_vector(page_vector, row, pre_row)
                if page == 'order':
                    product_vector = update_product_vector(product_vector, row)
                pre_page = page
                pre_gap = gap
                pre_row = row

        click_index=click_index+1


    with open('click_summary.pkl', 'wb') as f:
        pickle.dump(click_summary, f)


session_summary_new=pd.DataFrame(session_summary, columns=page_list + product_code_list + remaining_title)
click_summary_new=pd.DataFrame(click_summary, columns=page_list + product_code_list + click_remaining_title)


# Construct action space
action_space=['session_trans','builder_control','products_control']
action_space2=get_paction_space(click_summary_new)
action_space.extend(action_space2) # the first element in action space is ''
page_space=click_summary_new['page'].unique().tolist()



# Loop through two list above to construct the training and testing data
lag_1=8
pre_session_len=lag_1
pre_state_len=10
pre_session = deque(maxlen=pre_session_len)
pre_state = deque(maxlen=pre_state_len)
feature_batch=[]
testing_batch=[]


user_list=click_summary_new['userid'].unique()
train_test_split = random.binomial(n=1, p=0.8, size=len(user_list)) # split the user into training and testing set
train_user=user_list[train_test_split.astype(bool)]
test_user=user_list[(1-train_test_split).astype(bool)]

for index, row in click_summary_new.iterrows():
    user_id=row[-1]
    within_session_id=row[-2]
    session_id=row[-3]

    if (index+1)<len(click_summary_new):
        if within_session_id>1 and session_id>2 and user_id==click_summary_new.iloc[index+1]['userid'] and session_id==click_summary_new.iloc[index+1]['session_index']:
            # the current record
            action = action_space.index(row[-4])
            page = page_space.index(row[-5])
            click_interval = row[-6]
            reward = click_summary_new.iloc[index+1]['totalcents'] # reward is from next state
            click_num = sum(row[0:len(page_list)])
            sale_num = sum(row[len(page_list):(len(product_code_list) - 1)])
            state = row[0:(len(page_list) + len(product_code_list) - 1)].tolist()
            state.append(click_interval)  # add interval into the state

            # collect pre-session
            t_id = session_id # session id start from 2
            while t_id > 2 and len(pre_state) < pre_session_len:
                record = session_summary_new[(session_summary_new['userid'] == user_id) & (session_summary_new['session_index'] == (t_id - 1))]
                if not record.empty:
                    record = record.values.tolist()[0]
                    record = record[0:len(record) - 4]  # the last four elements are not state representation
                    pre_session.appendleft(record)  # appendleft ensure the temporal order
                t_id = t_id - 1
            pre_session_state = list(copy.deepcopy(pre_session))

            # collect pre-within_session state
            w_id = within_session_id
            while w_id > 0 and len(pre_state) < pre_state_len:
                record = click_summary_new[(click_summary_new['userid'] == user_id) & (click_summary_new['session_index'] == (session_id)) & (click_summary_new['within_session_index'] == (w_id))]
                if not record.empty:
                    record = record.values.tolist()[0]
                    temp_state = record[0:(len(page_list) + len(product_code_list) - 1)]
                    temp_state.append(record[-6])
                    pre_state.appendleft(temp_state)  # appendleft ensure the temporal order
                w_id = w_id - 1
            within_session_pre_state = list(copy.deepcopy(pre_state))


            # collect within_session state
            next_state=click_summary_new.iloc[index+1][0:(len(page_list) + len(product_code_list) - 1)].tolist()
            next_state.append(click_summary_new.iloc[index+1]['gap'])
            pre_state.append(next_state)
            within_session_state = list(copy.deepcopy(pre_state))
            next_page=page_space.index(click_summary_new.iloc[index+1]['page'])


            if user_id in train_user:
                # Add into the training memory
                if len(pre_session)>0 and len(pre_session)== lag_1 and len(pre_state)>1:
                    rand = np.random.rand()
                    if action > 2 or (action <= 2 and rand < 0.01) or (reward> 0) or (reward== 0 and rand < 0.01):
                        feature=[pre_session_state,within_session_pre_state,within_session_state,action,reward,page,next_page,user_id] # this is the feature will be used to train the model
                        feature_batch.append(feature)
            elif len(pre_session)== lag_1:
                feature = [pre_session_state, within_session_pre_state, within_session_state, action, reward, page,next_page,
                           user_id]  # this is the feature will be used to test the model
                testing_batch.append(feature)

            if len(feature_batch)>0 and len(feature_batch)%10000==0:
                with open('training_feature_batch.pkl', 'wb') as f:
                    pickle.dump(feature_batch, f)

    pre_state.clear()
    pre_session.clear()






with open('training_feature_batch.pkl', 'wb') as f:
    pickle.dump(feature_batch, f)

with open('testing_feature_batch.pkl', 'wb') as f:
    pickle.dump(testing_batch, f)