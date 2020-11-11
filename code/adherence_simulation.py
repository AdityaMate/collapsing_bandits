#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:50:43 2019

@author: adityamate, killian-34
"""

import numpy as np
import pandas as pd
import time
import pomdp

from itertools import combinations
from whittle import *
from utils import *
from generateTmatrix import *


import os
import argparse
import tqdm 

OPT_SIZE_LIMIT = 8


def takeAction(adherence, current_adherence, belief, actions, T, random_stream, T_hat=None):
    """
    
    """
    N=len(current_adherence)

    ###### Get next adhrence (ground truth)
    # Use the ground truth T here
    next_adherence=np.zeros(current_adherence.shape)
    for i in range(N):

        current_state=int(current_adherence[i])

        next_state=random_stream.binomial(1,T[i,int(actions[i]),current_state,1])

        next_adherence[i]=next_state


    ##### Update belief vector
    # Remember to use T_hat here
    for i in range(N):

        # belief = Prob(A)*Prob(A-->A) + Prob(NA)*Prob(NA-->A)
        if int(actions[i])==0:
            belief[i]= belief[i]* T_hat[i][0][1][1] + (1-belief[i])*(T_hat[i][0][0][1])

        elif int(actions[i])==1:
            belief[i]=current_adherence[i]* T_hat[i][1][1][1] + (1-current_adherence[i])*(T_hat[i][1][0][1])
            #   This relies on the assumption that on being called at least yesterday's
            #   adherence is perfectly observable. If not replace current_adherence[i] by belief[i]

    ##### Record observation
    observations=np.zeros(N)
    for i in range(N):
        if actions[i]==0:
            observations[i]=None
        else:
            observations[i]=current_adherence[i]

    return next_adherence, belief, observations


def getActions(N, k, belief=None, T_hat=None,policy_option=0,
               current_node=None, policy_graph_dict=None,
               days_since_called=None,last_observed_state=None,w=None,w_new=None,newWhittle=True,
               adherence_oracle=None, days_remaining=None, current_t=None,
               observations=None, adherence=None, T=None, verbose=False):

    """
    
    0: never call
    1: Call all patients everyday
    2: Randomly pick k patients to call
    3: Myopic policy
    4: pomdp
    5: whittle
    6: 2-day look ahead
    7: oracle
    8: whittle oracle
    9: DESPOT
    10: Yundi's whittle
    11: naive belief
    12: naive truth (for real, not belief)
    13: naive real proportion
    14: MDP whittle oracle
    15: Round Robin
    """
    if policy_option==0:
        ################## Nobody
        return np.zeros(N)

    elif policy_option==1:
        ################## Everybody
        return np.ones(N)

    elif policy_option==2:
        ################## Random
        # Randomly pick k arms out N arms
        random_call_indices=np.random.choice(N,size=k, replace=False)
        return np.array([1 if i in random_call_indices else 0 for i in range(N)])

    elif policy_option==3:
        ################## Myopic policy
        actions=np.zeros(N)
        myopic_rewards=np.zeros(N)
        for i in range(N):

            b=belief[i]  # Patient is adhering today with probability b

            b_next_nocall= b*(T_hat[i][0][1][1]) + (1-b)*(T_hat[i][0][0][1])

            b_next_call= b*(T_hat[i][1][1][1]) + (1-b)*(T_hat[i][1][0][1])

            myopic_rewards[i]=b_next_call-b_next_nocall
            # Myopic reward can be thought of as: Prob(A)*1 + Prob(NA)*0 = b

        #Pick the k greatest values from the array myopic_rewards
        patients_to_call=returnKGreatestIndices(myopic_rewards, k)

        for patient in patients_to_call:
            actions[patient]=1

        return actions

    elif policy_option==4:
        ################## optimal pomdp
        return policy_graph_dict[current_node][0]

    elif policy_option==5:
        ################## Whittle Index policy
        #Initialize if inputs not given
        if days_since_called.any() ==None:
            days_since_called=np.zeros(N) # Initialize to 0 days since last called (means nothing much)

        if  last_observed_state.any()==None:
            last_observed_state=np.ones(N) # Initialize to all patients found adhering last

        actions=actions=np.zeros(N)

        whittle_indices= [w[patient][int(last_observed_state[patient])][int(days_since_called[patient])] for patient in range(N)]
    
        patients_to_call=returnKGreatestIndices(whittle_indices, k)
        
        for patient in patients_to_call:
            actions[patient]=1

        return actions

    elif policy_option==6:
        ################## 2-day look ahead

        actions= list(combinations(range(N),k)) # TODO: list of NCk actions
        best_action=-1
        best_E_adherence=0
        for a1 in actions:
            # a1 is the list of k arms to pull today according to this action.
            # compute next-day belief vectors
            next_day_belief=np.zeros(N)
            for i in range(N):
                if i in a1:
                    next_day_belief[i]= belief[i]* T_hat[i][1][1][1] + (1-belief[i])*(T_hat[i][1][0][1])
                else:
                    next_day_belief[i]= belief[i]* T_hat[i][0][1][1] + (1-belief[i])*(T_hat[i][0][0][1])
            E_adherence1=np.sum(next_day_belief)

            # Compute next-day belief vectors

            a2=getActions(N,k, belief= next_day_belief,T_hat=T_hat,policy_option=3)

            next2_day_belief=np.zeros(N)
            for i in range(N):
                if i in a2:
                    next2_day_belief[i]= next_day_belief[i]* T_hat[i][1][1][1] + (1-next_day_belief[i])*(T_hat[i][1][0][1])
                else:
                    next2_day_belief[i]= next_day_belief[i]* T_hat[i][0][1][1] + (1-next_day_belief[i])*(T_hat[i][0][0][1])
            E_adherence2=np.sum(next2_day_belief)

            if E_adherence1+E_adherence2>best_E_adherence:
                best_action=a1
                best_E_adherence=E_adherence1+E_adherence2

        final_action=np.zeros(N)
        for patient in best_action:
            final_action[patient]=1
        return final_action

    elif policy_option==7:
        ################## oracle
        actions=getActions(N,k,belief=adherence_oracle,T_hat=T_hat,policy_option=3)
        return actions

    elif policy_option==8:
        ################## whittle oracle
        return pomdp.yundi_whittle_parallel(N, k, T_hat, adherence_oracle, days_remaining, mdp=False, n_processes=args.num_processes,verbose=verbose)

    ###################### 9 is despot

    ###################### Yundi's whittle index
    elif policy_option==10:
        return pomdp.yundi_whittle_parallel(N, k, T_hat, belief, days_remaining, mdp=False, n_processes=args.num_processes, verbose=verbose)

    ###################### naive belief
    elif policy_option==11:
        # yes i know it's bad coding practice to use global variables
        global belief_num_days_missed
        if 'belief_num_days_missed' not in globals():
            # initialize array if it doesn't exist
            belief_num_days_missed = np.zeros(N)

        # increment everything by one day
        belief_num_days_missed += 1

        # reset to 0 if we believe patient current adhering
        belief_num_days_missed[np.where(belief > .5)[0]] = 0


        # set observations -- ground truth
        observed_patients = ~np.isnan(observations)
        belief_num_days_missed[observed_patients] = observations[observed_patients]

        patients_to_call = (-belief_num_days_missed).argsort()[:k]
        actions = np.zeros(N)
        actions[patients_to_call] = 1

        return actions

    ###################### naive real
    elif policy_option==12:
        # yes i know it's bad coding practice to use global variables
        global real_num_days_missed
        if 'real_num_days_missed' not in globals():
            # initialize array if it doesn't exist
            real_num_days_missed = np.zeros(N)

        # increment everything by one day
        real_num_days_missed += 1

        # set adherence based on ground truth
        adhering_patients = np.where(adherence==1)[0]
        real_num_days_missed[adhering_patients] = 1

        patients_to_call = (-real_num_days_missed).argsort()[:k]
        actions = np.zeros(N)
        actions[patients_to_call] = 1

        return actions


    ###################### naive real porportional to transition probabilities
    elif policy_option==13:
        # yes i know it's bad coding practice to use global variables
        # global real_num_days_missed
        if 'real_num_days_missed' not in globals():
            # initialize array if it doesn't exist
            real_num_days_missed = np.zeros(N)

        # increment everything by one day
        real_num_days_missed += 1

        # set adherence based on ground truth
        adhering_patients = np.where(adherence==1)[0]
        real_num_days_missed[adhering_patients] = 0

        # num days missed * probability adhere if we call
        # plus probability stop adhering (no action), minus probability stop adhering if we call
        # priority = real_num_days_missed * T[:, 1, 0, 1] + T[:, 0, 1, 0] - T[:, 1, 1, 0]
        priority = real_num_days_missed * T[:, 0, 1, 0]
        #print('priority', priority)

        patients_to_call = (-priority).argsort()[:k]
        actions = np.zeros(N)
        actions[patients_to_call] = 1

        return actions

    elif policy_option==14:
        ################## whittle oracle
        return pomdp.yundi_whittle_parallel(N, k, T_hat, adherence_oracle, days_remaining, mdp=True, n_processes=args.num_processes,verbose=verbose)

    elif policy_option==15:
        ################## Round Robin
        active_arms= [(k*current_t+item)%N for item in range(k)]
        actions=[1 if i in active_arms else 0 for i in range(N)]
        return np.array(actions)
    
    elif policy_option==16:
        ################## New whittle index
        #Initialize if inputs not given
        if days_since_called.any() ==None:
            days_since_called=np.zeros(N) # Initialize to 0 days since last called (means nothing much)

        if  last_observed_state.any()==None:
            last_observed_state=np.ones(N) # Initialize to all patients found adhering last

        actions=actions=np.zeros(N)
        new_whittle_indices=np.zeros(N)
        for patient in range(N):
            if isBadPatient(T_hat[patient]) and newWhittle:
                w_bad=[[],[]]
                limits=[0,0]
                limits[int(last_observed_state[patient])]=days_since_called[patient]+1
                w_bad[1], w_bad[0]= newnewWhittle(T_hat[patient], version=-1, limit_a=limits[1], limit_na=limits[0])
                
                new_whittle_indices[patient]=(w_bad[int(last_observed_state[patient])])[int(days_since_called[patient])]
                
            else:
                if last_observed_state[patient]==0:
                    new_whittle_indices[patient]=fastWhittle(T_hat[patient], x2=days_since_called[patient]+1)
                else:
                    new_whittle_indices[patient]=fastWhittle(T_hat[patient], x1=days_since_called[patient]+1)
                
        patients_to_call=returnKGreatestIndices(new_whittle_indices, k)
        
        for patient in patients_to_call:
            actions[patient]=1

        return actions
    
    elif policy_option==17:
        ################## Fast whittle index
        #Initialize if inputs not given
        if days_since_called.any() ==None:
            days_since_called=np.zeros(N) # Initialize to 0 days since last called (means nothing much)

        if  last_observed_state.any()==None:
            last_observed_state=np.ones(N) # Initialize to all patients found adhering last

        actions=actions=np.zeros(N)
        fast_whittle_indices=np.zeros(N)
        for patient in range(N):
            if last_observed_state[patient]==0:
                fast_whittle_indices[patient]=fastWhittle(T_hat[patient], x2=days_since_called[patient]+1)
            else:
                fast_whittle_indices[patient]=fastWhittle(T_hat[patient], x1=days_since_called[patient]+1)
                
        patients_to_call=returnKGreatestIndices(fast_whittle_indices, k)
        
        for patient in patients_to_call:
            actions[patient]=1

        return actions
    elif policy_option==18: 
        ################## Buggy whittle index
        return getActions(N, k, belief=belief, T_hat=T_hat,policy_option=5,
               current_node=current_node, policy_graph_dict=policy_graph_dict,
               days_since_called=days_since_called,last_observed_state=last_observed_state,
               w=w,w_new=w_new,newWhittle=newWhittle,
               adherence_oracle=adherence_oracle, days_remaining=days_remaining, current_t=current_t,
               observations=observations, adherence=adherence, T=T, verbose=verbose)
    


def learnTmatrixFromObservations(observations, actions, random_stream):
    '''
    observations and actions are L+1 and L-sized matrices with:
        Observations: [o0, o1,...oL] with each entry being 0=NA;  1=A
        Actions:      [a1, a2,...aL] with each entry being 0=NoCall; 1=Called
    '''
    T=np.zeros((2,2,2))
    p_pass_01, p_pass_11, p_act_01, p_act_11=sorted(random_stream.uniform(size=4))
    l=len(actions)
    vals, counts = np.unique(list(zip(observations[:l], actions, observations[1:])), axis=0, return_counts=True)
    
    freq=np.zeros((2,2,2))
    
    for i, item in enumerate(vals):
        
        freq[int(item[0]),int(item[1]),int(item[2])]=counts[i]
        
    if (freq[0,0,0]+freq[0,0,1])>0:
        p_pass_01 = freq[0,0,1]/(freq[0,0,0]+freq[0,0,1])
    
    if (freq[1,0,0]+freq[1,0,1])>0:
        p_pass_11 = freq[1,0,1]/(freq[1,0,0]+freq[1,0,1])
    
    if (freq[0,1,0]+freq[0,1,1])>0:
        p_act_01 = freq[0,1,1]/(freq[0,1,0]+freq[0,1,1])
    
    if (freq[1,1,0]+freq[1,1,1])>0:
        p_act_11 = freq[1,1,1]/(freq[1,1,0]+freq[1,1,1])

    T[0]=np.array([[1-p_pass_01, p_pass_01],[1-p_pass_11, p_pass_11]])
    T[1]=np.array([[1-p_act_01, p_act_01],[1-p_act_11, p_act_11]])
            
        
    return T
    
def update_counts(adherence, actions, last_called, current_round, counts, buffer_length=0, get_last_call_transition_flag=False):
	'''
	# update counts
	# get all information about a patient since the last time we called
	# the transition (last_called[i], last_called[i]+1) will be the only info we get about a T1 matrix
	# all others will update our info about T0
	# one assumption we make here is that patient adherence is determined in the morning before we call
	# but in practice this isn't true -- this method would lead to degenerate counts in that case
	# but probably not a big issue
	'''

	if buffer_length == 0:
		buffer_length = 100000000

	# Buffer is how much patient "remembers" which doesn't include today.
	# so add 1 to the buffer_length to make code cleaner below, i.e. adding
	# 1 makes the buffer include today.
	buffer_length+=1

	patients_called = [i for i,a in enumerate(actions) if a==1]

	for i in patients_called:
		info_packet = adherence[i, last_called[i]:current_round+1].astype(int)

		curr = None
		prev = None

		# if it doesn't fit in the buffer cut it, but conditionally add the t1
		# remaining adds will be to t0
		if info_packet.shape[0] > buffer_length:

			if get_last_call_transition_flag:
				prev = info_packet[0]
				curr = info_packet[1]
				counts[i, 1, prev, curr] += 1

			info_packet = info_packet[-buffer_length:]
			prev = info_packet[0]
			curr = info_packet[1]
			counts[i, 0, prev, curr] += 1
			prev = curr

		# Else first add will be to t1
		else:
			prev = info_packet[0]
			curr = info_packet[1]
			counts[i, 1, prev, curr] += 1
			prev = curr

		# The rest is about T0
		for j in range(2, len(info_packet)):
			curr = info_packet[j]
			counts[i, 0, prev, curr] += 1
			prev = curr

		# record that we called this patient
		last_called[i] = current_round



def thompson_sampling(N, priors, counts, random_stream):

    T_hat = np.zeros((N,2,2,2))
    for i in range(N):
        for j in range(T_hat.shape[1]):
            for k in range(T_hat.shape[2]):
                params = priors[i, j, k, :] + counts[i, j, k, :]
                T_hat[i, j, k, :] = random_stream.dirichlet(params)
    return T_hat

def thompson_sampling_constrained(N, priors, counts, random_stream):

    T_hat = np.zeros((N,2,2,2))
    for i in range(N):
    	# While sampled T_hat is not valid or has not been sampled yet...
    	while (not verify_T_matrix(T_hat[i]) or T_hat[i].sum() == 0):
	        for j in range(T_hat.shape[1]):
	            for k in range(T_hat.shape[2]):
	                params = priors[i, j, k, :] + counts[i, j, k, :]
	                T_hat[i, j, k, :] = random_stream.dirichlet(params)
    return T_hat



def simulateAdherence(N,L,T,k, policy_option, start_node=None, policy_graph_dict=None,
                        obs_space=None, action_logs={}, cum_adherence=None, 
                        new_whittle=True, online=True,
                        seedbase=None, savestring='trial', epsilon=0.0, learning_mode=False, 
                        world_random_seed=None, learning_random_seed=None, verbose=False,
                        buffer_length=0, get_last_call_transition_flag=False, file_root=None):
    """
    """
    
    
    learning_random_stream = np.random.RandomState()
    if learning_mode > 0:
        learning_random_stream.seed(learning_random_seed)

    world_random_stream = np.random.RandomState()
    world_random_stream.seed(world_random_seed)

    T_hat = None
    if learning_mode == 2:
        T_hat = generateRandomTmatrix(N, random_stream=learning_random_stream)    
    priors = np.ones((N,2,2,2))
    counts = np.zeros((N,2,2,2))
    last_called = np.zeros(N).astype(int)

    if learning_mode == 4:
    	T_hat = computeAverageTmatrixFromData(N, file_root=file_root)

    adherence=np.zeros((N,L))
    actions_record=np.zeros((N, L-1))

    if action_logs is not None:
        action_logs[policy_option] = []

    adherence[:,0]=np.ones(N)
    belief=np.ones(N)

    current_node = None

    w=None
    w_new=None
    #############################
    #######  pomdp policy #######
    #############################
    if policy_option==4 and N < OPT_SIZE_LIMIT:
        initial_belief = np.zeros(2**N)
        initial_belief[-1] = 1.0
        fname, a, obs_space = pomdp.make_pomdp(N, k, T, O=None, discount=0.95, method='exact', root=args.file_root)
        start_node, policy_graph_dict = pomdp.solve(fname, initial_belief, a, root=args.file_root)
        current_node = start_node
        print("running OPT")

    #############################
    #######  Whittle Index ######
    #############################
    if ((policy_option==5 or policy_option==16 or policy_option==18) and (not online) and (not learning_mode)):
        # Pre-compute only if policy is 5 or 16 AND it is neither online nor learning case. 
        print("Pre-computing whittle index for offline, no-learning mode")
        # Pre-compute whittle index for patients
        w=np.zeros((N, 2, L))
        w_new=np.zeros((N, 2, L)) # right now, w_new does not get used in takeAction() even though it's passed in.
        for patient in range(N):
            if policy_option==5:
                w[patient,1,:], w[patient,0,:]= whittleIndex(T[patient], L=L)
            if policy_option==18:
                w[patient,0,:], w[patient,1,:]= whittleIndex(T[patient], L=L)
            
    # Keep track of days since called and last observed state
    days_since_called=np.zeros(N) # Initialize to 0 days since called
    last_observed_state=np.ones(N)

    #######  Run simulation #######
    print('Running simulation w/ policy: %s'%policy_option)
    # make array of nan to initialize observations
    observations = np.full(N, np.nan)
    learning_modes=['no_learning', 'Thompson sampling', 'e-greedy','Constrained TS','Naive Mean']
    print("Learning mode:", learning_modes[learning_mode])
        
    epsilon_schedule = [epsilon]*(L-1) # always explore with epsilon
    if epsilon == 0.0: # else anneal epsilon from 1 to 0.0.
        # power = np.log(y)/np.log(1-x)
        # where y = desired epsilon when we are x% of the way through treatment
        # so if we want epsilon to be 0.25 by the time we are 25% of the way through treatment
        # we get: np.log(0.25)/np.log(1-0.25) = 4.818841679306418
        power = 4.818841679306418
        epsilon_schedule = np.linspace(1,0.00,L)**power 
        # Note that we never have epsilon 0 since we never access the last element.

    for t in tqdm.tqdm(range(1,L)):
        
        '''
        Learning T_hat from simulation so far
        '''
        #print("Round: %s"%t)
        days_remaining = L-t
        if learning_mode==0:
            T_hat=T
        elif learning_mode == 1:
            # Thompson sampling
            T_hat = thompson_sampling(N, priors, counts, random_stream=learning_random_stream)            
        elif learning_mode==2 and t>2:
            # Epsilon-Greedy
            for patient_number, action in enumerate(actions):# Note that actions here is still the previous day's action record                
                if action==1:
                    T_hat[patient_number]=learnTmatrixFromObservations(adherence[patient_number, :t-1],
                        actions_record[patient_number, 1:(t-1)], random_stream=learning_random_stream)
        
        elif learning_mode == 3:
            # Thompson sampling
            T_hat = thompson_sampling_constrained(N, priors, counts, random_stream=learning_random_stream)

        elif learning_mode == 4:
        	# Naive average baseline
        	pass


        EPSILON_CLIP=0.0005
        T_hat= epsilon_clip(T_hat, EPSILON_CLIP)
            
        if online or learning_mode:
            # If neither online nor learning, then just work with pre-computed whittle indices, w and w_new.
            w=np.zeros((N, 2, L))
            w_new=np.zeros((N, 2, L))
            
            if policy_option==5:
                
                for patient in range(N):
                    limits=[0,0]
                    limits[int(last_observed_state[patient])]=days_since_called[patient]+1
                    w[patient,1,:], w[patient,0,:]= whittleIndex(T_hat[patient], L=L, limit_a=limits[1], limit_na=limits[0])

            if policy_option==16:
                pass
                
            if policy_option==18:
                
                for patient in range(N):
                    limits=[0,0]
                    limits[int(last_observed_state[patient])]=days_since_called[patient]+1
                    w[patient,0,:], w[patient,1,:]= whittleIndex(T_hat[patient], L=L, limit_a=limits[1], limit_na=limits[0])
        
        #### Epsilon greedy part
        
        if learning_mode == 2 and (policy_option not in NON_EPSILON_POLICIES): # epsilon greedy
            if learning_random_stream.binomial(1,epsilon_schedule[t])==0: #Exploitation
                actions=getActions(N, k, policy_option=policy_option, belief=belief, T_hat=T_hat,
                               current_node=current_node, policy_graph_dict = policy_graph_dict,
                               days_since_called=days_since_called,
                               last_observed_state=last_observed_state, w=w,w_new=w_new,current_t=t,
                               adherence_oracle=adherence[:,t-1].squeeze(), days_remaining=days_remaining,
                               observations=observations, adherence=adherence[:, t-1], T=T,
                               verbose=verbose)
            else:  # Exploration
                actions=getActions(N, k, policy_option=2, belief=belief, T_hat=T_hat,
                                   current_node=current_node, policy_graph_dict = policy_graph_dict,
                                   days_since_called=days_since_called,
                                   last_observed_state=last_observed_state, w=w,w_new=w_new,current_t=t,
                                   adherence_oracle=adherence[:,t-1].squeeze(), days_remaining=days_remaining,
                                   observations=observations, adherence=adherence[:, t-1], T=T,
                                   verbose=verbose)
        else: # Normal process
            actions=getActions(N, k, policy_option=policy_option, belief=belief, T_hat=T_hat,
               current_node=current_node, policy_graph_dict = policy_graph_dict,
               days_since_called=days_since_called,
               last_observed_state=last_observed_state, w=w,w_new=w_new, current_t=t,
               adherence_oracle=adherence[:,t-1].squeeze(), days_remaining=days_remaining,
               observations=observations, adherence=adherence[:, t-1], T=T,
               verbose=verbose)
            
                
        actions_record[:, t-1]=actions    
        
        if action_logs is not None:
            action_logs[policy_option].append(actions.astype(int))


        adherence[:,t], belief, observations=takeAction(adherence, adherence[:,t-1].squeeze(), belief, actions,T, random_stream=world_random_stream, T_hat=T_hat)

        # update counts
        # get all information about a patient since the last time we called
        # the transition (last_called[i], last_called[i]+1) will be the only info we get about a T1 matrix
        # all others will update our info about T0
        update_counts(adherence, actions, last_called, t, counts, buffer_length=buffer_length, get_last_call_transition_flag=get_last_call_transition_flag)

        ###### Update last observed state and last called matrix:
        for i in range(N):
            if actions[i]==0:
                days_since_called[i]+=1
            else:
                days_since_called[i]=0
                last_observed_state[i]=observations[i]


        if policy_option==4:

            observation = np.zeros(N)
            patients_observed = [i for i,a in enumerate(actions) if a == 1]
            observation[patients_observed] = adherence[[patients_observed], t]+1
            observation = tuple(observation.astype(int))

            observation_index = obs_space.index(observation)
            current_node = policy_graph_dict[current_node][1][observation_index]


    if cum_adherence is not None:
        cum_adherence[policy_option] = np.cumsum(adherence.sum(axis=0))


    return adherence




if __name__=="__main__":

    """
    0: never call    1: Call all patients everyday     2: Randomly pick k patients to call
    3: Myopic policy    4: pomdp  5: whittle    6: 2-day look ahead    7:oracle
    8: oracle_whittle   9: despot 10: yundi's whittle index
    11: naive belief (longest since taking a pill belief)
    12: naive real (longest since taking a pill ground truth)
    13: naive real, multiplied by ground truth transition probability
    """



    parser = argparse.ArgumentParser(description='Run adherence simulations with various POMDP methods.')
    parser.add_argument('-n', '--num_patients', default=2, type=int, help='Number of Patients')
    parser.add_argument('-k', '--num_calls_per_day', default=1, type=float, help='Number of calls per day')
    parser.add_argument('-l', '--simulation_length', default=180, type=int, help='Number of days to run simulation')
    parser.add_argument('-N', '--num_trials', default=5, type=int, help='Number of trials to run')
    parser.add_argument('-d', '--data', default='real', choices=['real','simulated','full_random','unit_test','myopic_fail', 'demo', 'uniform'], type=str,help='Method for generating transition probabilities')
    parser.add_argument('-s', '--seed_base', type=int, help='Base for the random seed')
    parser.add_argument('-ws','--world_seed_base', default=None, type=int, help='Base for the random seed')
    parser.add_argument('-ls','--learning_seed_base', default=None, type=int, help='Base for the random seed')
    parser.add_argument('-p', '--num_processes', default=4, type=int, help='Number of cores for parallelization')
    parser.add_argument('-f', '--file_root', default='./..', type=str,help='Root dir for experiment (should be the dir containing this script)')
    parser.add_argument('-pc', '--policy', default=-1, type=int, help='policy to run, default is all policies')
    parser.add_argument('-res', '--results_file', default='answer', type=str, help='adherence numpy matrix file name')
    parser.add_argument('-tr', '--trial_number', default=None, type=int, help='Trial number')
    parser.add_argument('-sv', '--save_string', default='', type=str, help='special string to include in saved file name')
    parser.add_argument('-badf', '--bad_fraction', default=0.4, type=float, help='fraction of non-responsive patients')
    parser.add_argument('-thrf_perc', '--threshopt_percentage', default=None, type=int, help='% of threshold optimal patients in data')
    parser.add_argument('-beta', '--beta', default=0.5, type=float, help='beta used in quick check for determining threshold optimal fraction')
    parser.add_argument('-ep', '--epsilon', default=0, type=float, help='espilon value for epsilon greedy')
    parser.add_argument('-lr', '--learning_option', default=0, choices=[0,1,2,3,4], type=int, help='0: No Learning (Ground truth known)\n1: Thompson Sampling\n2 Epsilon Greedy\n3 Constrained TS\n4 Naive average baseline')
    parser.add_argument('-v', '--verbose', default=False, type=bool)
    parser.add_argument('-o', '--online', default=0, type=int, help='0: offline, 1: online')
    parser.add_argument('-kp', '--k_percentage', default=None, type=int, help='100* k/N ')
    parser.add_argument('-slurm', '--slurm_array_id', default=-1, type=int, help='Unique identifier for slurm array id/ encoding set of parameters')    
    parser.add_argument('-sh1', '--shift1', default=0.05, type=float, help='shift 1 variable ')
    parser.add_argument('-sh2', '--shift2', default=0.05, type=float, help='shift 2 variable ')
    parser.add_argument('-sh3', '--shift3', default=0.05, type=float, help='shift 3 variable ')
    parser.add_argument('-sh4', '--shift4', default=0.05, type=float, help='shift 4 variable ')
    parser.add_argument('-bl', '--buffer_length', default=0, type=int, help='If using Thompson Sampling, max number of most recent days of adherence you learn with an arm pull')
    parser.add_argument('-t1f', '--get_last_call_transition_flag', default=0, type=int, help='If using Thompson Sampling, whether or not you learn the T1 transition regardless of buffer_length with an arm pull')
    args = parser.parse_args()

    """
    POLICY NAMES ***
    0: never call    1: Call all patients everyday     2: Randomly pick k patients to call
    3: Myopic policy    4: pomdp  5: Threshold whittle    6: 2-day look ahead    7:oracle
    8: oracle_whittle   9: despot 10: yundi's whittle index 11,12,13: Lily 
    14: MDP oracle     15: round robin  16: new new whittle(fast)  17: fast whittle 18: Buggy whittle
    """
    NON_EPSILON_POLICIES = [0, 1, 14, 15]
    
    if args.slurm_array_id>=0:
        """
        Code for SLURM
        """
        '''
        Changing tr: 0-49, policy: {10,14}, N:{10,20,100,200,500,1000,2000}
        '''
        
        slurm_trial_nums=[i for i in range(50)]
        slurm_policies=[10,14]
        slurm_N=[200,500,1000,2000]
        #slurm_th_fracs=[0,10,20,30,40,50,60,70,80,90,100]
        
        args.trial_number=args.slurm_array_id%len(slurm_trial_nums)
        args.policy= slurm_policies[int(args.slurm_array_id//len(slurm_trial_nums))%len(slurm_policies)]
        args.num_patients=slurm_N[int(args.slurm_array_id//(len(slurm_trial_nums)*len(slurm_policies)))%len(slurm_N)]
        #args.threshopt_percentage=slurm_th_fracs[int(args.slurm_array_id//(len(slurm_trial_nums)*len(slurm_policies)*len(slurm_N)))%len(slurm_th_fracs)]
        #args.save_string+=("_threshopt_frac_"+str(args.threshopt_percentage))
    
    
    
    ##### File root
    if args.file_root == '.':
        args.file_root = os.getcwd()
    ##### k
    args.num_calls_per_day = int(args.num_calls_per_day)
    if args.k_percentage is not None:
        args.num_calls_per_day = int((args.k_percentage/100 * args.num_patients)) # This rounds down, good.
    ##### Save special name
    if args.save_string=='':
        args.save_string=str(time.ctime().replace(' ', '_').replace(':','_'))
    else:
        args.save_string=args.save_string
    ##### Policies to run
    if args.policy<0:
        #policies = [0, 1, 2, 3, 5, 8, 10]
        # policies = [0, 1, 2, 3, 5, 8, 10, 14]
        # policies = [0, 1, 2, 3, 5, 10, 14, 15,16] # initial experiment of learning
        # policies = [0, 1, 2, 3, 5, 8, 10, 14, 15,16, 17, 18] # initial experiment of learning

        policies = [0,1,2,3,5, 10, 14] # initial experiment of learning
        #policies= [0,1,2,3,5, 10,14]

    else:
        policies=[args.policy]
    ##### Seed = seed_base + trial_number
    if args.trial_number is not None:
        args.num_trials=1
        add_to_seed_for_specific_trial=args.trial_number
    else:
        add_to_seed_for_specific_trial=0
    first_seedbase=np.random.randint(0, high=100000)
    if args.seed_base is not None:
        first_seedbase = args.seed_base+add_to_seed_for_specific_trial

    first_world_seedbase=np.random.randint(0, high=100000)
    if args.world_seed_base is not None:
        first_world_seedbase = args.world_seed_base+add_to_seed_for_specific_trial

    first_learning_seedbase=np.random.randint(0, high=100000)
    if args.learning_seed_base is not None:
        first_learning_seedbase = args.learning_seed_base+add_to_seed_for_specific_trial

    ##### Other parameters
    N=args.num_patients
    L=args.simulation_length
    k=args.num_calls_per_day
    savestring=args.save_string
    N_TRIALS=args.num_trials
    LEARNING_MODE=args.learning_option
    #LEARNING_MODE='EpsilonGreedy'#'False'
    #LEARNING_MODE='Thompson'#'False'
    #LEARNING_MODE='False'
    
    record_policy_actions=[3, 4, 5, 6, 11, 12, 13, 7, 8, 10, 14, 15,16, 17, 18]
    # size_limits: run policy if N< size_limit; ALso size_limit=-1 means all N are ok. Size_limit=0 means switched off.
    size_limits={ 0:None, 1:None, 2: None, 3: None, 4:OPT_SIZE_LIMIT, 5: None
                ,6:4 ,    7:0,     8:1000,  9:0,     10:None,
                11: None, 12: None, 13: None, 14: None, 15: None, 16:None, 17:None, 18:None}

    # policy names dict
    pname={0: 'nobody',     1: 'everyday',     2: 'Random',
           3: 'Myopic',     4: 'optimal',      5: 'Threshold whittle',
           6: '2-day',      7: 'oracl_m',      8: 'oracle_POMDP',
           9: 'despot',     10: 'yundi',       11: 'naiveBelief',
           12: 'naiveReal', 13: 'naiveReal2',  14: 'oracle_MDP',
           15: 'Round Robin', 16:'New_whittle', 17: 'FastWhittle' ,
           18: 'BuggyWhittle'}


    adherences=[[] for i in range(len(pname))]
    adherence_matrices=[None for i in range(len(pname))]
    action_logs = {}
    cum_adherence = {}

    start=time.time()
    file_root=args.file_root
    
    
    
    
    
    for i in range(N_TRIALS):


        seedbase = first_seedbase + i
        np.random.seed(seed=seedbase)

        world_seed_base = first_world_seedbase + i
        learning_seed_base = first_learning_seedbase + i

        #print (args.seed_base)
        print ("Seed is", seedbase)
        #print (args.online)
        T = None
        if args.data == 'real':
            
            if args.threshopt_percentage is not None:
                T=generateTmatrixReal(N, file_root=args.file_root, 
                                      thresh_opt_frac=(args.threshopt_percentage)/100., beta=args.beta, quick_check=False)
            else:
                T=generateTmatrixReal(N, file_root=args.file_root, 
                                      thresh_opt_frac=None, beta=args.beta, quick_check=False)
            '''Temporary code for perturbation experiments'''
            """
            T=generateTmatrixReal(N, file_root=args.file_root,
                        shift1=args.shift1,shift2=args.shift2,
                        shift3=args.shift3,shift4=args.shift4)
            """
            #print(how_many_NIB(T));raise ValueError

        elif args.data == 'simulated':
            # T = generateTmatrix(N, responsive_patient_fraction=1.- args.bad_fraction)
            T = generateTmatrixBadf(N, responsive_patient_fraction=1.- args.bad_fraction)
        elif args.data == 'full_random':
            # T = generateTmatrix(N, responsive_patient_fraction=1.- args.bad_fraction)
            T = generateTmatrixFullRandom(N, badf=args.bad_fraction)
        elif args.data =='unit_test':
            T = unit_testing.identical_patients(N)
        elif args.data =='myopic_fail':
            T = generateYundiMyopicFailTmatrix()
        elif args.data =='demo':
            T = specialTmatrix(N, option=1, badf=args.bad_fraction)
        elif args.data =='uniform':
            T = generateTmatrixNIBandIB(N, thresh_opt_frac=(args.threshopt_percentage)/100., beta=args.beta, quick_check=False)
            
        #print ("T is  as follows:")
        #print (T)
        #print ("Printed T")
        #np.save("seed0", T)
        np.random.seed(seed=seedbase)
            #N = 2
            #k=1

        for policy_option in policies:
            #print (policy_option)
            ############################ Policy # policy_option
            policy_start_time=time.time()
            if size_limits[policy_option]==None or size_limits[policy_option]>N:
                np.random.seed(seed=seedbase)
                if policy_option in record_policy_actions:
                    adherence_matrix=simulateAdherence(N,L,T,k,policy_option=policy_option, seedbase=seedbase, action_logs=action_logs, 
                                                       cum_adherence=cum_adherence, epsilon=args.epsilon, learning_mode=LEARNING_MODE, 
                                                       learning_random_seed=learning_seed_base, world_random_seed=world_seed_base,
                                                       verbose=args.verbose, online=(args.online==1), buffer_length=args.buffer_length,
                                                       get_last_call_transition_flag=args.get_last_call_transition_flag, file_root=file_root)
                    adherence_matrices[policy_option]= adherence_matrix
                    #np.save(file_root+'/adherence_log/rebuttal/adherence_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s'%(N,k,L,policy_option,args.data,args.bad_fraction,seedbase, LEARNING_MODE), adherence_matrix)
                    np.save(file_root+'/logs/adherence_log/adherence_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s'%(N,k,L,policy_option,args.data,args.bad_fraction,seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag), adherence_matrix)
                    adherences[policy_option].append(np.mean(np.sum(adherence_matrix, axis=1)))

                else:
                    if args.verbose:
                        print(learning_seed_base,'LRSEED\n\n\n\n\n')
                        print(world_seed_base,'LRSEED\n\n\n\n\n')
                    adherence_matrix=simulateAdherence(N,L,T,k,policy_option=policy_option, seedbase=seedbase, learning_mode=LEARNING_MODE, 
                                                       learning_random_seed=learning_seed_base, world_random_seed=world_seed_base,
                                                       verbose=args.verbose,online=(args.online==1), buffer_length=args.buffer_length,
                                                       get_last_call_transition_flag=args.get_last_call_transition_flag, file_root=file_root)
                    adherence_matrices[policy_option]= adherence_matrix
                    #np.save(file_root+'/adherence_log/rebuttal/adherence_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s'%(N,k,L,policy_option,args.data,args.bad_fraction, seedbase, LEARNING_MODE), adherence_matrix)
                    np.save(file_root+'/logs/adherence_log/adherence_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s'%(N,k,L,policy_option,args.data,args.bad_fraction, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag), adherence_matrix)
                    adherences[policy_option].append(np.mean(np.sum(adherence_matrix, axis=1)))

            else:
                adherence_matrix=np.zeros((N,L))
                adherence_matrices[policy_option]= adherence_matrix
                adherences[policy_option]= np.mean(np.sum(adherence_matrix, axis=1))
            policy_end_time=time.time()
            policy_run_time=policy_end_time-policy_start_time
            #np.save(file_root+'/runtime/rebuttal/runtime_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s'%(N,k,L,policy_option,args.data,args.bad_fraction,seedbase, LEARNING_MODE), policy_run_time)
            np.save(file_root+'/logs/runtime/runtime_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s'%(N,k,L,policy_option,args.data,args.bad_fraction,seedbase, LEARNING_MODE, args.buffer_length,args.get_last_call_transition_flag), policy_run_time)
        ##### SAVE ALL RELEVANT LOGS #####

        # write out action logs
        for policy_option in action_logs.keys():
            fname = os.path.join(args.file_root,'logs/action_logs/action_logs_'+savestring+'_N%s_k%s_L%s_data%s_badf%s_policy%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,L, args.data,args.bad_fraction, policy_option, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
            columns = list(map(str, np.arange(N)))
            df = pd.DataFrame(action_logs[policy_option], columns=columns)
            df.to_csv(fname, index=False)

        # write out cumulative adherence logs
        for policy_option in cum_adherence.keys():
            fname = os.path.join(args.file_root,'logs/cum_adherence/cum_adherence_'+savestring+'_N%s_k%s_L%s_data%s_badf%s_policy%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,L, args.data,args.bad_fraction, policy_option, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
            columns = list(map(str, np.arange(L)))
            df = pd.DataFrame([cum_adherence[policy_option]], columns=columns)
            df.to_csv(fname, index=False)

        # write out T matrix logs
        fname = os.path.join(args.file_root,'logs/Tmatrix_logs/Tmatrix_logs_'+savestring+'_N%s_k%s_L%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,L, args.data,args.bad_fraction, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
        np.save(fname, T)


    for p in range(max(policies) + 1):
        print (pname[p],": ", np.mean(adherences[p]))

    end=time.time()
    print ("Time taken: ", end-start)

    if args.policy<0 and False:
        '''
        Default option (old code copy pasted under if)for running all policies code and all old stuff.
        '''
        
        # policies_to_plot = [0,2,15,3,6,5,10,9,4,8,14,1]
        # [0, 1, 2, 3, 5, 10, 14, 15,16]
        policies_to_plot = [0,2,15,3,6,16,5,18,10,14,1]
        policies_to_plot=[0,2,15,3, 5, 4]
        bottom=0
        labels = [pname[i] for i in policies_to_plot]
        values=[round(np.mean(np.array(adherences[i]))-bottom, 1) for i in policies_to_plot]
        errors=[np.std(np.array(adherences[i])) for i in policies_to_plot]
        #labels = ['Nobody', 'k Random', 'k Myopic', '2-day', 'Whittle', 'Yundi', 'DESPOT', 'Optimal','Oracle', 'Everybody']
        #values = [round(np.mean(adherence[0]),1), round(np.mean(adherence[2]),1), round(np.mean(adherence[3]),1), round(np.mean(adherence[6]),1),round(np.mean(adherence[5]),1), round(np.mean(adherence[10]),1), round(np.mean(adherence[9]),1), round(np.mean(adherence[4]),1),round(np.mean(adherence[8]),1),round(np.mean(adherence[1]),1)]
        #errors = [np.std(adherence0), np.std(adherence2), np.std(adherence3), np.std(adherence6),np.std(adherence5), np.std(adherence10), np.std(adherence9), np.std(adherence4),np.std(adherence8), np.std(adherence1)]

        vals = [values, errors]
        df = pd.DataFrame(vals, columns=labels)
        fname = os.path.join(args.file_root,'logs/results/results'+savestring+'_N%s_k%s_trials%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,N_TRIALS, args.data,args.bad_fraction, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
        df.to_csv(fname,index=False)
        
        
        '''Convert values to percentage'''
        percentages= [round(100*(values[i]-values[0])/(values[5]-values[0]),0) for i in range(len(values))]
        values=percentages[1:]
        labels=labels[1:]
        errors=errors[1:]
        barPlot(labels, values, errors, ylabel='Intervention benefit as %',
            title='%s patients, %s calls per day; trials: %s ' % (N, k, N_TRIALS),
            filename=file_root+'/img/results_'+savestring+'_N%s_k%s_trials%s_data%s_s%s_lr%s.png'%(N,k,N_TRIALS, args.data,first_seedbase, LEARNING_MODE), root=args.file_root,
            bottom=0)
