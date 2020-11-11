#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:24:48 2020

@author: adityamate, killian-34
"""
import numpy as np
import pandas as pd
import time
import pomdp

from itertools import combinations
from whittle import *
from utils import *


import os
import argparse
import tqdm 


def computeAverageTmatrixFromData(N, file_root='.', epsilon=0.005):

	"""
	Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
	action=0 denotes passive action, a=1 is active action
	State 0 denotes NA and state 1 denotes A
	"""
	fname = os.path.join(file_root, 'data/patient_T_matrices.npy')
	real = np.load(fname)

	T=np.zeros((N,2,2,2))
	#Passive action transition probabilities
	penalty_pass_00=0
	penalty_pass_11=0

	#Active action transition probabilities
	benefit_act_00=0
	benefit_act_11=0


	avg = real.mean(axis=0)

	# for i in range(N):

	T_base = np.zeros((2,2))
	T_base[0,0] = avg[0]
	T_base[1,1] = avg[1]
	T_base[0,1] = 1 - T_base[0,0]
	T_base[1,0] = 1 - T_base[1,1]

	T_base = smooth_real_probs(T_base, epsilon)

	shift = 0.05

	# Patient responds well to call
	benefit_act_00=np.random.uniform(low=0., high=shift) # will subtract from prob of staying 0,0
	benefit_act_11= benefit_act_00 + np.random.uniform(low=0., high=shift) # will add to prob of staying 1,1
	# add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition

	# Patient does well on their own, low penalty for not calling
	penalty_pass_11=np.random.uniform(low=0., high=shift) # will sub from prob of staying 1,1
	penalty_pass_00=penalty_pass_11+np.random.uniform(low=0., high=shift) # will add to prob of staying 0,0
	    


	T_pass = np.copy(T_base)
	T_act = np.copy(T_base)

	T_act[0,0] = max(0, T_act[0,0] - benefit_act_00)
	T_act[1,1] = min(1, T_act[1,1] + benefit_act_11)

	T_pass[0,0] = min(1, T_pass[0,0] + penalty_pass_00)
	T_pass[1,1] = max(0, T_pass[1,1] - penalty_pass_11)

	T_pass[0,1] = 1 - T_pass[0,0]
	T_pass[1,0] = 1 - T_pass[1,1]

	T_act[0,1] = 1 - T_act[0,0]
	T_act[1,0] = 1 - T_act[1,1]

	T_pass = epsilon_clip(T_pass, epsilon)
	T_act = epsilon_clip(T_act, epsilon)

	#print(T_pass)
	#print(T_act)
	#print()



	if not verify_T_matrix(np.array([T_pass, T_act])):
	    print("T matrix invalid\n",np.array([T_pass, T_act]))
	    raise ValueError()

	for i in range(N):
	    T[i,0]=T_pass
	    T[i,1]=T_act


	return T



# See page 7 of:
# https://projects.iq.harvard.edu/files/teamcore/files/2016_15_teamcore_aamas2016_eve_yundi.pdf

def specialTmatrix(N, kfrac=10, distribution=[0.5, 0.5], delta=0.02, option=2, badf=50):
    
    option =3
    if option==0:
        
        T=np.zeros((N,2,2,2))
        
        patient_descriptions=[]
        T_p_01=[0.3, 0.3]
        T_p_11=[0.97, 0.1]
        T_a_01=[0.3, 0.9]
        T_a_11=[0.97, 0.97]
        
        
        for i in range(N):
            
            index=np.random.choice(range(len(distribution)), p=distribution)
            T[i][0][0][1]=np.random.uniform(T_p_01[index]-delta, T_p_01[index]+delta)
            T[i][0][1][1]=np.random.uniform(T_p_11[index]-delta, T_p_11[index]+delta)
            T[i][1][0][1]=np.random.uniform(T_a_01[index]-delta, T_a_01[index]+delta)
            T[i][1][1][1]=np.random.uniform(T_a_11[index]-delta, T_a_11[index]+delta)
            
        return T
    
    elif option==1:
        
        T=np.zeros((N,2,2,2))
        
        k=int(kfrac*N/100.)
        
        # Myopic wants to pull type 2
        '''
        type1 = np.array( [[[0.9,  0.1],
                 [0.6,  0.41]],
    
                [[0.6, 0.4],
                 [0.3, 0.7]]])
        

        type2 = np.array( [[[0.9,  0.1],
                 [0.6,  0.4]],
    
                [[0.6, 0.4],
                 [0.3, 0.7]]])    
        '''
        
        type1 = np.array( [[[0.6,  0.4],
                 [0.29,  0.71]],
    
                [[0.35, 0.65],
                 [0.05, 0.95]]])
        

        type2 = np.array( [[[0.6,  0.4],
                 [0.3,  0.7]],
    
                [[0.35, 0.65],
                 [0.05, 0.95]]])
    
    
        for i in range(k):
            T[i] = type2
        
        for j in range(k, N):
            type1 = np.array( [[[0.6,  0.4],
                 [0.29,  0.71+ j*0.001]],
    
                [[0.35, 0.65],
                 [0.05, 0.95]]])
            T[j]=type1
            
        
        print ("Returning T matrix: ")
        print ("N: ", N, "k: ", k)
        print ("shape: ", T.shape)
        return T


    elif option==2:
        
        T=np.zeros((N,2,2,2))

        type1= [[[0.97, 0.03],
                  [0.03, 0.97]],

                [[0.96, 0.04],
                 [0.01, 0.99]]]
       
        type2 = [[[0.25, 0.75],
                    [0.03, 0.97]],

                [[0.23, 0.77],
                 [0.01     , 0.99     ]]]

        T[0]=type1
        T[1]=type2
    
        return T
       
    elif option==3: 
        
        
        shift1= 0.05
        shift2= 0.05
        shift3= 0.05
        shift4= 0.05
        epsilon=0.01
        T=np.zeros((N,2,2,2))

        type1= [[[0.97, 0.03],
                  [0.03, 0.97]],

                [[0.96, 0.04],
                 [0.01, 0.99]]]          ###### Bad patient
       
        type2 = [[[0.25, 0.75],
                    [0.03, 0.97]],

                [[0.23, 0.77],
                 [0.01     , 0.99     ]]]   ##### Good patient (self-healing)
        
        for i in range(N):
            
            types=[type1, type2]
            type_choice=types[np.random.choice([0, 1],p=[badf/100., 1-(badf/100.)])]
            T[i]=np.array(type_choice)
            
            
            # add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition
            benefit_act_00=np.random.uniform(low=0., high=shift1) # will subtract from prob of staying 0,0
            benefit_act_11= benefit_act_00 + np.random.uniform(low=0., high=shift2) # will add to prob of staying 1,1
            # Patient does well on their own, low penalty for not calling
            penalty_pass_11=np.random.uniform(low=0., high=shift3) # will sub from prob of staying 1,1
            penalty_pass_00=penalty_pass_11+np.random.uniform(low=0., high=shift4) # will add to prob of staying 0,0


            T[i][1][0][0]= max(0, T[i][1][0][0] - benefit_act_00)            
            T[i][1][1][1]= min(1, T[i][1][1][1] + benefit_act_11)            
            
            T[i][0][0][0]= min(1, T[i][0][0][0] + penalty_pass_00)            
            T[i][0][1][1]= max(0, T[i][0][1][1] - penalty_pass_11)  
            
            T[i][0][0][1]=   1- T[i][0][0][0]          
            T[i][0][1][0]=   1- T[i][0][1][1]
            
            T[i][1][0][1]=   1- T[i][1][0][0]          
            T[i][1][1][0]=   1- T[i][1][1][1]
            
            T[i][0]=epsilon_clip(T[i][0], epsilon)
            T[i][1]=epsilon_clip(T[i][1], epsilon)
            
        return T
            
    
def generateYundiMyopicFailTmatrix():

    
    # Return a randomly generated T matrix (not unformly random because of sorting)
    T=np.zeros((2,2,2,2))
    # T[0] = [[[0.95, 0.05],
    #          [0.05, 0.95]],

    #         [[0.99, 0.01],
    #          [0.1,  0.9]]]

    # T[1] = [[[0.4, 0.6],
    #          [0.1, 0.9]],

    #         [[0.7,  0.3],
    #          [0.4,  0.6]]]

    T[0] =  [[[0.99, 0.01],
             [0.1,  0.9]],

            [[0.95, 0.05],
             [0.05, 0.95]]]

    T[1] =  [[[0.7,  0.3],
             [0.4,  0.6]],

            [[0.4, 0.6],
             [0.1, 0.9]]]



    return T 

def generateRandomTmatrix(N, random_stream):

    # Return a randomly generated T matrix (not unformly random because of sorting)
    T=np.zeros((N,2,2,2))
    for i in range(N):
        p_pass_01, p_pass_11, p_act_01, p_act_11=sorted(random_stream.uniform(size=4))
        T[i,0]=np.array([[1-p_pass_01, p_pass_01],[1-p_pass_11, p_pass_11]])
        T[i,1]=np.array([[1-p_act_01, p_act_01],[1-p_act_11, p_act_11]])
    return T  


def generateTmatrix(N, responsive_patient_fraction=0.4, 
                    range_pass_00=(0.8,1.0), range_pass_11=(0.6,0.9), 
                    range_act_g_00=(0,0.2),range_act_g_11=(0.9,1.0), 
                    range_act_b_00=(0.6,0.8), range_act_b_11=(0.9,1.0)):

    # p_act01 < p01/(p01+p10)


    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
 
        
    T=np.zeros((N,2,2,2))
    #Passive action transition probabilities
    p_pass_00=np.random.uniform(low=range_pass_00[0], high=range_pass_00[1], size=N)
    p_pass_11=np.random.uniform(low=range_pass_11[0], high=range_pass_11[1], size=N)

    #Active action transition probabilities
    #responsive_patient_fraction=0.4
    p_act_00=np.zeros(N)
    p_act_11=np.zeros(N)
    for i in range(N):
        if np.random.binomial(1,responsive_patient_fraction)==1:
            # Patient responds well to call
            p_act_00[i]=np.random.uniform(low=range_act_g_00[0], high=range_act_g_00[1])
            p_act_11[i]=np.random.uniform(low=range_act_g_11[0], high=range_act_g_11[1])
        else:
            # Patient doesn't respond well to call
            p_act_00[i]=np.random.uniform(low=range_act_b_00[0], high=range_act_b_00[1])
            p_act_11[i]=np.random.uniform(low=range_act_b_11[0], high=range_act_b_11[1])


    for i in range(N):
        T[i,0]=np.array([[p_pass_00[i], 1-p_pass_00[i]],[1-p_pass_11[i],p_pass_11[i]]])
        T[i,1]=np.array([[p_act_00[i], 1-p_act_00[i]],[1-p_act_11[i],p_act_11[i]]])

    #print (T[:20])
    return T

# guaranteed to generate 'bad patients' according to the definition here:
# p_act01 < p01/(p01+p10) == bad
# as well as good patients according to the same.
# we only want to consider bottom chain bad patients because top chain bad patients
# would mean our action has negative effect on them which isn't realistic.
# but this gives bad separation from myopic
def generateTmatrixBadf(N, responsive_patient_fraction=0.4, 
                    range_pass_00=(0.6,0.8), range_pass_11=(0.6,0.89), 
                    range_act_g_00=(0,0.2),range_act_g_11=(0.9,1.0), 
                    range_act_b_00=(0.7,0.9), range_act_b_11=(0.9,1.0)):

    # print("p_act01 < p01/(p01+p10)")
    

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
 
        
    T=np.zeros((N,2,2,2))
    #Passive action transition probabilities
    p_pass_00=np.random.uniform(low=range_pass_00[0], high=range_pass_00[1], size=N)
    p_pass_11=np.random.uniform(low=range_pass_11[0], high=range_pass_11[1], size=N)

    #Active action transition probabilities
    #responsive_patient_fraction=0.4
    p_act_00=np.zeros(N)
    p_act_11=np.zeros(N)
    for i in range(N):
        if np.random.binomial(1,responsive_patient_fraction)==1:
            # Patient responds well to call
            p_act_00[i]=np.random.uniform(low=range_act_g_00[0], high=range_act_g_00[1])
            p_act_11[i]=np.random.uniform(low=range_act_g_11[0], high=range_act_g_11[1])

            p_act01 = 1-p_act_00[i]
            p01 = 1-p_pass_00[i]
            p10 = 1-p_pass_11[i]
            if p_act01 < p01/(p01+p10):
                raise ValueError("Intended good patient was bad.")
        else:
            # Patient doesn't respond well to call
            p_act_00[i]=np.random.uniform(low=range_act_b_00[0], high=range_act_b_00[1])
            p_act_11[i]=np.random.uniform(low=range_act_b_11[0], high=range_act_b_11[1])

            p_act01 = 1-p_act_00[i]
            p01 = 1-p_pass_00[i]
            p10 = 1-p_pass_11[i]
            if not (p_act01 < p01/(p01+p10)):
                raise ValueError("Intended bad patient was good.")
        
        




    for i in range(N):
        T[i,0]=np.array([[p_pass_00[i], 1-p_pass_00[i]],[1-p_pass_11[i],p_pass_11[i]]])
        T[i,1]=np.array([[p_act_00[i], 1-p_act_00[i]],[1-p_act_11[i],p_act_11[i]]])

    #print (T[:20])
    return T


# guaranteed to generate 'bad patients' according to the definition here:
# p_act01 < p01/(p01+p10) == bad
# as well as good patients according to the same.
# we only want to consider bottom chain bad patients because top chain bad patients
# would mean our action has negative effect on them which isn't realistic.
# but this gives bad separation from myopic
def generateTmatrixFullRandom(N,badf=0.2):

    # print("p_act01 < p01/(p01+p10)")
    

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
 
        
    T=np.zeros((N,2,2,2))
    
    for i in range(N):

        should_be_bad_patient = np.random.binomial(1,badf)==1

        valid = False
        while not valid:

            this_T = np.random.dirichlet([1,1],size=(2,2))
            if should_be_bad_patient:

                p_act01 = this_T[1][0][1]
                p01 = this_T[0][0][1]
                p10 = this_T[0][1][0]

                is_bad_patient = p_act01 < p01/(p01+p10)

                is_valid_matrix = verify_T_matrix(this_T)

                valid = is_bad_patient and is_valid_matrix

            else:

                p_act01 = this_T[1][0][1]
                p01 = this_T[0][0][1]
                p10 = this_T[0][1][0]

                is_bad_patient = p_act01 < p01/(p01+p10)

                is_valid_matrix = verify_T_matrix(this_T)

                valid = (not is_bad_patient) and is_valid_matrix
        
        if should_be_bad_patient != (p_act01 < p01/(p01+p10)):
            raise ValueError("Mismatch")
        
        T[i] = this_T
        

    # print (T)
    # 1/0
    return T

def generateTmatrixNIBandIB(N,thresh_opt_frac=1, beta=0.5, quick_check=False):

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
        
    T=np.zeros((N,2,2,2))
    thres_opt_patients=np.random.choice([i for i in range(N)],size=int(thresh_opt_frac*N), replace=False)
    for i in range(N):

        valid = False
        while not valid:
            this_T = np.random.dirichlet([1,1],size=(2,2))
            valid = verify_T_matrix(this_T)
            if valid and thresh_opt_frac is not None: 
                satisfies_condition=False
                if i in thres_opt_patients:                 # Threshold opt patient
                    satisfies_condition=isThresholdOptimal(this_T,beta, quick_check=quick_check)
                else:                                       # Reverse Threshold opt patient
                    satisfies_condition=isReverseThresholdOptimal(this_T,beta, quick_check=quick_check)
                valid=satisfies_condition     
            
        T[i] = this_T
        

    # print (T)
    # 1/0
    return T



def generateTmatrixNIBandIBFast(N):

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
        
    T=np.zeros((N,2,2,2))
    for i in range(N):

        valid = False
        while not valid:
            this_T = np.random.dirichlet([1,1],size=(2,2))
            valid = verify_T_matrix(this_T) 
            
        T[i] = this_T
        

    # print (T)
    # 1/0
    return T


# there are only 41 of 8350 cases where 
# p11 < p10 results from not (p11=0.0 or p10=1.0)
def smooth_real_probs(T, epsilon):


    # T = epsilon_clip(T, epsilon)

    if T[1,1] < T[0,1]:

        # make p11 and p01 equal so we can properly simulate
        # action effects

        # If it looks like this, make t01 = t11
        # [[0.0,  1.0],
        #  [0.01, 0.99]]]

        # If it looks like this, make t11 = t01
        # [[0.95, 0.05],
        #  [1.0,  0.0]]]

        if T[0,1] >= 0.5:
            T[0,1] = T[1,1]
        else:
            T[1,1] = T[0,1]

        T[0,0] = 1- T[0,1]
        T[1,0] = 1- T[1,1]

    return T




def generateTmatrixReal(N, file_root='.', responsive_patient_fraction=0.4, epsilon=0.005,
                        shift1=0,shift2=0,shift3=0,shift4=0, intervention_effect=0.05, 
                        thresh_opt_frac=None, beta=0.5, quick_check=False):

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
    fname = os.path.join(file_root+'/data/', 'patient_T_matrices.npy')
    real = np.load(fname)

    T=np.zeros((N,2,2,2))
    #Passive action transition probabilities
    penalty_pass_00=0
    penalty_pass_11=0

    #Active action transition probabilities
    benefit_act_00=0
    benefit_act_11=0

    if thresh_opt_frac is None:
        choices = np.random.choice(np.arange(real.shape[0]), N, replace=True)
    else:
        thres_opt_patients=np.random.choice([i for i in range(N)],size=int(thresh_opt_frac*N), replace=False)
        
    i=0
    while i < N:
        
        if thresh_opt_frac is None:
            choice = choices[i]
        else:
            choice=np.random.choice(np.arange(real.shape[0]), 1, replace=True)[0]
        T_base = np.zeros((2,2))
        T_base[0,0] = real[choice][0]
        T_base[1,1] = real[choice][1]
        T_base[0,1] = 1 - T_base[0,0]
        T_base[1,0] = 1 - T_base[1,1]

        T_base = smooth_real_probs(T_base, epsilon)


        shift = intervention_effect

        # Patient responds well to call
        benefit_act_00=np.random.uniform(low=0., high=shift) # will subtract from prob of staying 0,0
        benefit_act_11= benefit_act_00 + np.random.uniform(low=0., high=shift) # will add to prob of staying 1,1
        # add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition
                
        
        # Patient does well on their own, low penalty for not calling
        penalty_pass_11=np.random.uniform(low=0., high=shift) # will sub from prob of staying 1,1
        penalty_pass_00=penalty_pass_11+np.random.uniform(low=0., high=shift) # will add to prob of staying 0,0
        
        
        '''
        For perturbation experiment only. TEMPORARY CODE below. 
        '''
        """
        benefit_act_00=np.random.uniform(low=0., high=shift1) # will subtract from prob of staying 0,0
        benefit_act_11= benefit_act_00 + np.random.uniform(low=0., high=shift2) # will add to prob of staying 1,1
        # add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition
                
        
        # Patient does well on their own, low penalty for not calling
        penalty_pass_11=np.random.uniform(low=0., high=shift3) # will sub from prob of staying 1,1
        penalty_pass_00=penalty_pass_11+np.random.uniform(low=0., high=shift4) # will add to prob of staying 0,0
        """
        
        

        T_pass = np.copy(T_base)
        T_act = np.copy(T_base)

        T_act[0,0] = max(0, T_act[0,0] - benefit_act_00)
        T_act[1,1] = min(1, T_act[1,1] + benefit_act_11)

        T_pass[0,0] = min(1, T_pass[0,0] + penalty_pass_00)
        T_pass[1,1] = max(0, T_pass[1,1] - penalty_pass_11)

        T_pass[0,1] = 1 - T_pass[0,0]
        T_pass[1,0] = 1 - T_pass[1,1]

        T_act[0,1] = 1 - T_act[0,0]
        T_act[1,0] = 1 - T_act[1,1]

        T_pass = epsilon_clip(T_pass, epsilon)
        T_act = epsilon_clip(T_act, epsilon)

        #print(T_pass)
        #print(T_act)
        #print()



        if not verify_T_matrix(np.array([T_pass, T_act])):
            print("T matrix invalid\n",np.array([T_pass, T_act]))
            raise ValueError()

        if thresh_opt_frac is None: 
            satisfies_condition=True
        
        else:
            satisfies_condition=False
            if i in thres_opt_patients:                 # Threshold opt patient
                satisfies_condition=isThresholdOptimal([T_pass,T_act],beta, quick_check=quick_check)
            else:                                       # Reverse Threshold opt patient
                satisfies_condition=isReverseThresholdOptimal([T_pass,T_act],beta, quick_check=quick_check)
        if satisfies_condition:
            T[i,0]=T_pass
            T[i,1]=T_act          
            i+=1
    return T



def generateTmatrixRealNoReplace(N, file_root='.', epsilon=0.005,
                        shift1=0,shift2=0,shift3=0,shift4=0, intervention_effect=0.05):

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
    fname = os.path.join(file_root, 'data/patient_T_matrices.npy')
    real = np.load(fname)

    T=np.zeros((N,2,2,2))
    #Passive action transition probabilities
    penalty_pass_00=0
    penalty_pass_11=0

    #Active action transition probabilities
    benefit_act_00=0
    benefit_act_11=0

    choices = np.random.choice(np.arange(real.shape[0]), N, replace=False)
        
    for i,choice in enumerate(choices):
        
        T_base = np.zeros((2,2))
        T_base[0,0] = real[choice][0]
        T_base[1,1] = real[choice][1]
        T_base[0,1] = 1 - T_base[0,0]
        T_base[1,0] = 1 - T_base[1,1]

        T_base = smooth_real_probs(T_base, epsilon)


        shift = intervention_effect

        # Patient responds well to call
        benefit_act_00=np.random.uniform(low=0., high=shift) # will subtract from prob of staying 0,0
        benefit_act_11= benefit_act_00 + np.random.uniform(low=0., high=shift) # will add to prob of staying 1,1
        # add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition
                
        
        # Patient does well on their own, low penalty for not calling
        penalty_pass_11=np.random.uniform(low=0., high=shift) # will sub from prob of staying 1,1
        penalty_pass_00=penalty_pass_11+np.random.uniform(low=0., high=shift) # will add to prob of staying 0,0
        
        
        '''
        For perturbation experiment only. TEMPORARY CODE below. 
        '''
        """
        benefit_act_00=np.random.uniform(low=0., high=shift1) # will subtract from prob of staying 0,0
        benefit_act_11= benefit_act_00 + np.random.uniform(low=0., high=shift2) # will add to prob of staying 1,1
        # add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition
                
        
        # Patient does well on their own, low penalty for not calling
        penalty_pass_11=np.random.uniform(low=0., high=shift3) # will sub from prob of staying 1,1
        penalty_pass_00=penalty_pass_11+np.random.uniform(low=0., high=shift4) # will add to prob of staying 0,0
        """
        
        

        T_pass = np.copy(T_base)
        T_act = np.copy(T_base)

        T_act[0,0] = max(0, T_act[0,0] - benefit_act_00)
        T_act[1,1] = min(1, T_act[1,1] + benefit_act_11)

        T_pass[0,0] = min(1, T_pass[0,0] + penalty_pass_00)
        T_pass[1,1] = max(0, T_pass[1,1] - penalty_pass_11)

        T_pass[0,1] = 1 - T_pass[0,0]
        T_pass[1,0] = 1 - T_pass[1,1]

        T_act[0,1] = 1 - T_act[0,0]
        T_act[1,0] = 1 - T_act[1,1]

        T_pass = epsilon_clip(T_pass, epsilon)
        T_act = epsilon_clip(T_act, epsilon)

        #print(T_pass)
        #print(T_act)
        #print()



        if not verify_T_matrix(np.array([T_pass, T_act])):
            print("T matrix invalid\n",np.array([T_pass, T_act]))
            raise ValueError()

        T[i,0]=T_pass
        T[i,1]=T_act

    return T