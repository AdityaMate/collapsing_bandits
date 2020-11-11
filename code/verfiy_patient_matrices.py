import numpy as np 
import os

def verfiy_patient_matrices():

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
    fname = os.path.join('.', 'patient_T_matrices.npy')
    real = np.load(fname)

    N = real.shape[0]

    choices = np.random.choice(np.arange(real.shape[0]), N, replace=False)
    T_0_set = set()
    T_1_set = set()
    T_0_set.add(None)
    T_1_set.add(None)
    inval_count = 0
    for i in range(N):
        T_0_tup = None
        T_1_tup = None


        choice = choices[i]
        T_base = np.zeros((2,2))
        T_base[0,0] = real[choice][0]
        T_base[1,1] = real[choice][1]
        T_base[0,1] = 1 - T_base[0,0]
        T_base[1,0] = 1 - T_base[1,1]

        if T_base[1,1] < T_base[0,1]:
        	if not(T_base[0,1] == 1.0 or T_base[1,1] == 0.0):
	        	inval_count+=1
	        	print (T_base[1,1], T_base[0,1], T_base[1,1] - T_base[0,1])
    print (inval_count,N)

verfiy_patient_matrices()