import mdptoolbox
from numba import jit
from tqdm import tqdm
#from whittle import precomputeBelief


import numpy as np

def tau(b):
    return p11*b+(1-b)*p01
def precomputeBelief(p11, p01, p11active, p01active,  L=180):
    #precomputeBelief(T[0][1][1],T[0][0][1],T[1][1][1],T[1][0][1])
    bA=np.zeros(L)
    bNA=np.zeros(L)
    
    bA[-1]=1.
    bNA[-1]=0.
    
    for t in range(L):  
        if t==0:
            bA[t]= bA[t-1]*p11active + (1.- bA[t-1])*p01active
            bNA[t]= bNA[t-1]*p11active + (1.- bNA[t-1])*p01active
        else:            
            bA[t]= bA[t-1]*p11 + (1.- bA[t-1])*p01
            bNA[t]= bNA[t-1]*p11 + (1.- bNA[t-1])*p01
    return bA, bNA    

@jit(nopython=True)
def mdp_setup(n_states, states, L, m):

    bs_T = np.zeros((2, n_states, n_states))
    bs_r = np.zeros((2, n_states, n_states))

    # top chain
    for i in range(len(states[0])):
        val = states[0][i]
        
        # no action
        if i < L - 1:
            bs_T[0][i][i+1] = 1
        else:
            bs_T[0][i][i] = 1

        # action
        # reset to head of top chain
        bs_T[1][i][0] = val
        # reset to head of bottom chain
        bs_T[1][i][L] = 1 - val

        bs_r[:,i,:] = val

    # bottom chain
    for i in range(len(states[1])):
        val = states[1][i]
        
        ind = i + L
        # no action
        if ind < 2*L - 1:
            bs_T[0][ind][ind+1] = 1
        else:
            bs_T[0][ind][ind] = 1

        # action
        # reset to head of top chain
        bs_T[1][ind][0] = val
        # reset to head of bottom chain
        bs_T[1][ind][L] = 1 - val

        bs_r[:,ind,:] = val

    # add the subsidy for passivity
    bs_r[0,:,:] += m

    return bs_T, bs_r

@jit(nopython=True)
def check_all_passive(m, V, state_list, beta):

    V_a_head = V[0] 
    V_p_head = V[len(V)//2]

    all_passive = True

    # Check A chain
    for i in range(0, len(V)//2):
        b = state_list[i]
        V_next = V[min(i+1,len(V)//2-1)] # On the last day of the chain, V_next = V_now
        all_passive &= m + beta*V_next > beta*(b*V_a_head + (1 - b)*V_p_head)
    
    # Check NA chain
    for i in range(len(V)//2, len(V)):
        b = state_list[i]
        V_next = V[min(i+1,len(V)-1)] # On the last day of the chain, V_next = V_now
        all_passive &= m + beta*V_next > beta*(b*V_a_head + (1 - b)*V_p_head)

    return all_passive



def brute_force_check_threshold_policy_type(T, check_forward=False, check_backward=False, L=180, num_ms=500, beta=0.5):
    
    if check_forward == check_backward:
        raise ValueError("Must set check_forward xor check_backward")    

    p11, p01 = T[0][1][1], T[0][0][1]
    p11active, p01active = T[1][1][1], T[1][0][1]

    # avoid_overflow(p11, p01, p11active, p01active)
    epsilon = 0.001
    if p01active == p01:
        p01 -= epsilon
    if p11active == p11:
        p11 -= epsilon

    states = precomputeBelief(p11, p01, p11active, p01active, L=L)

    n_states = 2*L

    # plt = plotBeliefTrajectorySimple(p11, p01, p11active, p01active)
    # plt.savefig('threshold_optimality_work/%s/belief_trajectory.png'%working_dir)
    # plt.show()

    states_concat = np.concatenate(states)
    belief_sort_indices = np.argsort(states_concat)[::-1]
    sorted_beliefs = states_concat[belief_sort_indices]

    b_stat = states_concat[-1]


    m_list = np.linspace(0, 2, num_ms)
    # beta_list = np.linspace(0.5, 0.99, 10)[::-1][:1]

    passed_all_m = True
    
    for m_ind, m in enumerate(m_list):


        fit_this_m = False

        bs_T, bs_r = mdp_setup(n_states, states, L, m)

        # Make mdp
        mdp = mdptoolbox.mdp.ValueIteration(bs_T, bs_r, discount=beta, epsilon=1e-6, max_iter=10000)
        # mdp = mdptoolbox.mdp.FiniteHorizon(bs_T, bs_r, discount=beta, N=L)
        mdp.run()

        V = np.array(mdp.V)
        policy = np.array(mdp.policy)
        V_sort_indices = np.argsort(V)[::-1]
        # V_sort_indices = np.argsort(V[:,0])[::-1]

        # chain_head_diff = V[0,0] - V[len(V)//2,0]

        V_a_head = V[0] 
        V_p_head = V[len(V)//2]

        b_min = min(b_stat, p01active)
        if b_stat <= p01active:
            V_min_next = V[-1]
        else:
            V_min_next = V[len(V)//2+1]


        V_a_head_next = V[1]

        # active at head
        b_head = states_concat[0]
        all_active =  m + beta*V_a_head_next < beta*(b_head*V_a_head + (1 - b_head)*V_p_head)
        # active at lowest belief
        all_active &= m + beta*V_min_next < beta*(b_min*V_a_head + (1 - b_min)*V_p_head)

        if passed_all_m:

            head_condition = None
            tail_condition = None

            if check_forward:
                # Forward threshold optimal
                head_condition = m + beta*V_a_head_next > beta*(b_head*V_a_head + (1 - b_head)*V_p_head)
                tail_condition = m + beta*V_min_next < beta*(b_min*V_a_head + (1 - b_min)*V_p_head)
            elif check_backward:
                # head active tail passive
                head_condition = m + beta*V_a_head_next < beta*(b_head*V_a_head + (1 - b_head)*V_p_head)
                tail_condition = m + beta*V_min_next > beta*(b_min*V_a_head + (1 - b_min)*V_p_head)

            fits_condition = tail_condition and head_condition
            if fits_condition:
                # break and report the beta that worked
                fit_this_m = True
            
            # If the above fails, check if all value functions are passive or all active
            elif all_active:
                # break and report the beta that worked
                fit_this_m = True

            elif check_all_passive(m, V, states_concat, beta):
                fit_this_m = True


        if not fit_this_m:
            passed_all_m = False
            break
    
    return passed_all_m







