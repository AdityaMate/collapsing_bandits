from __future__ import print_function
import numpy as np 
import argparse
from itertools import product, combinations
from functools import reduce
from tqdm import tqdm
import subprocess
import os
import multiprocessing
import platform
# import special_pomdp

import traceback, functools, multiprocessing
  
def trace_unhandled_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print ('Exception in '+func.__name__)
            traceback.print_exc()
    return wrapped_func


# path constants
pomdp_file_folder = 'pomdp_files'
output_folder = 'pg'

# Defaults, probably won't work though.
despot_solver_path = 'pomdpx'
pomdp_solve_path = 'pomdp-solve'
# CPLEX_PATH = '/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx'

if platform.release() == '3.10.0-957.12.1.el7.x86_64': # cannon
	despot_solver_path = '/n/home10/jkillian/.local/bin/despot'
	pomdp_solve_path = '/n/home10/jkillian/.local/bin/pomdp-solve'

elif platform.release() == '4.15.0-64-generic': # deathstar
	despot_solver_path = '/home/jkillian/.local/bin/despot'
	pomdp_solve_path = '/home/jkillian/.local/bin/pomdp-solve'
	
elif platform.release() == '18.6.0': # Jack Macbook 2019
	despot_solver_path = 'pomdpx'
	pomdp_solve_path = 'pomdp-solve'
	# CPLEX_PATH = '/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx'

if not os.path.exists(pomdp_file_folder):
    os.makedirs(pomdp_file_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# @trace_unhandled_exceptions
def yundi_special_worker(args):
    """This function will be called by each thread.
    This function can not be a class method.
    """
    # Expand list of args into named args.
    mid, candidate, T, belief, ind, days_remaining = args
    del args

    if candidate:
        # fname, a, obs_space = make_pomdp_given_cost(mid, T, ind, discount=1.)
        # print(T)
        initial_belief = np.array([1-belief, belief])
        # solverPath = CPLEX_PATH
        # optimal_action = optimal_strategy(mid, days_remaining, solverPath, T, initial_belief)
        optimal_action = optimal_strategy(mid, days_remaining, T, initial_belief)
        optimal_results[ind] = optimal_action

        if (optimal_action == 1):
            # Serial-only Portion
            with trueCount.get_lock():
                trueCount.value += 1
        

def optimal_strategy(subsidy, roundLeft, T, belief):

	num_states = T.shape[0]
	bestAction = POMDP_solve(subsidy, roundLeft, T)

	result = -1;
	maxSum = -(2**31)
	for temp in bestAction.keys():
		sum = 0
		for i in range(num_states):
			sum += temp[i] * belief[i];
		
		if (sum > maxSum):
			maxSum = sum
			result = bestAction[temp]
		
	

	if (result == 1):
		return 1;
	elif result == 0:
		return 0;
	else:
		print('error!')
		return 0
	
	

def POMDP_solve(subsidy, roundLeft, T, beta=0.9):

	n = 1

	num_states = 2
	num_observations = 2
	num_actions = 2


	O1 = [
		[1.0, 0.0],
		[0.0, 1.0]
	]


	PomdpO = np.zeros((num_states, num_actions, num_observations + 1));
	PomdpR = np.zeros((num_states, num_actions));

	for state in range(num_states):
		PomdpO[state][0][num_observations] = 1;
		PomdpO[state][1][num_observations] = 0;
		for observation in range(num_observations):
			PomdpO[state][0][observation] = 0;
			PomdpO[state][1][observation] = O1[state][observation];
		
	

	for state in range(num_states):
		PomdpR[state][0] = subsidy+state;
		PomdpR[state][1] = state;
	# print('pomdpr',PomdpR)
		
	
	epsilon = 0.2;
	# print('hey')
	pomdp_object = special_pomdp.pomdpSolver(num_states, num_observations + 1, num_actions,
			beta, T, PomdpO, PomdpR, roundLeft, epsilon);
	pomdp_object.solve();
	# // pomdp.solverSolve(solverPath);
	return pomdp_object.bestAction;


def tup_print(tups, joiner='', f=None):
    tup_strs = [joiner.join(list(map(make_char, tup))) for tup in tups]
    # print(' '.join(tup_strs),file=f)
    return tup_strs

def make_char(x):
    return chr(x+65)

def yundi_whittle_exact(T, belief, beta, solver='normal'):

    upper = 2
    lower = 0

    optimal_action = 0

    gap = 10000
    gap_tol = 1e-4 
    while gap > gap_tol:

        mid = (upper + lower) / 2.0;
        #print('mid',mid)
        trueCount = 0
        # Spawn up to 9999999 jobs, I think this is the maximum possible.
        # I do not know what happens if you exceed this.
        if solver == 'normal':

            fname, a, obs_space = make_pomdp_given_cost(mid, T, 0, discount=beta)
            # print(T)
            initial_belief = belief
            if True:
                initial_belief = np.array([1-belief, belief])

            optimal_action = solve_given_cost(fname, initial_belief, a)
            
            if optimal_action == 0: 
                upper = mid;

            else:
                lower = mid
        gap = upper - lower

    
    return (upper + lower) / 2.0




def print_POMDP_given_cost(combined_action_space, combined_state_space, combined_observation_space,
		T_matrices, O_matrices, R, C,
		 pomdp_filename, discount):
	
	# fname = os.path.join(root,pomdp_file_folder)
	# fname = os.path.join(fname, pomdp_filename)
	fname = os.path.join(pomdp_file_folder, pomdp_filename)
	fout = open(fname, 'w')

	print('discount: %.2f'%discount,file=fout)
	print('values: reward',file=fout)
	print('actions: ',end='', file=fout)
	action_space_strs = tup_print(combined_action_space)
	print(' '.join(action_space_strs),file=fout)
	print('states: ', end='', file=fout); 
	state_space_strs = tup_print(combined_state_space)
	print(' '.join(state_space_strs),file=fout)
	print('observations: ', end='', file=fout); 
	observation_space_strs = tup_print(combined_observation_space)
	print(' '.join(observation_space_strs),file=fout)
	print(file=fout)

	for i, action in enumerate(action_space_strs):
		print('T:%s'%action,file=fout)
		for row in T_matrices[i]:
			print(' '.join(list(map(str, row))), file=fout)
		print(file=fout)

	for i, action in enumerate(action_space_strs):
		print('O:%s'%action,file=fout)
		for row in O_matrices[i]:
			print(' '.join(list(map(str, row))), file=fout)
		print(file=fout)

	for i,state in enumerate(state_space_strs):
		for j, action in enumerate(action_space_strs):
		# print('R:* : * : %s : * %i' % (state, R[i]), file=fout)
			r = R[i]
			# If we don't call, we get C subsidy
			if j == 0:
				r = R[i] + C
			print('R:%s : %s : * : * %.4f' % (action, state, r), file=fout)
			# R: <action> : <start-state> : <end-state> : <observation> %f

	fout.close()


def make_pomdp_given_cost(C, T, ind, discount=0.95):

	n = 1

	num_states = 2
	num_observations = 3


	states_per_patient = np.arange(num_states)
	combined_state_space = list(product(states_per_patient, repeat=n))
	# print('State Space')
	# print(combined_state_space)
	# print()

	patient_indices = np.arange(n)
	combined_action_space = list(combinations(patient_indices, 1))
	combined_action_space = [(0,),(1,)]

	# print('Action Space')
	# print(combined_action_space)
	# print()

	observations_per_patient = np.arange(num_observations)
	combined_observation_space = list(product(observations_per_patient, repeat=n))
	

	T_matrices = T # first one should be no action, second is action


	O0 = [
		[1.0, 0.0, 0.0],
		[1.0, 0.0, 0.0]
	]

	O1 = [
		[0.0, 1.0, 0.0],
		[0.0, 0.0, 1.0]
	]

	d = {
		0: O0,
		1: O1
	}

	O_matrices = [O0, O1]




	# R: <action> : <start-state> : <end-state> : <observation> %f
	# so do 
	# and compute for all end states such that it is the sum of the 1's in the end state
	# R: * : * : <end-state> : * %f
	R = [sum(x) for x in combined_state_space]


	pomdp_filename = 'single_patient_pomdp_patient=%s_c=%s.POMDP' %(ind,C)
	print_POMDP_given_cost(combined_action_space, combined_state_space, combined_observation_space,
		T_matrices, O_matrices, R, C, pomdp_filename = pomdp_filename, 
		discount=discount)
	return pomdp_filename, combined_action_space, combined_observation_space


def solve_given_cost(fname, initial_belief, action_space):

	outname = os.path.join(output_folder, fname)
	# outname = os.path.join(root, outname)

	fname = os.path.join(pomdp_file_folder, fname)
	# fname = os.path.join(root, fname)

	subprocess.check_output([pomdp_solve_path, '-pomdp', fname, '-o', outname])

	alpha_fname = outname+'.alpha'
	pg_fname = outname+'.pg'

	pg_d = {}
	pg_f = open(pg_fname,'r')
	for line in pg_f:
		line = line.strip().split()
		# print(line)
		# ['0', '1', '-', '19', '19', '-', '-', '-', '-', '-', '-']
		node_num = int(line[0])

		action = int(line[1])
		
		obs_list = line[2:]
		obs_list = [int(x) if x!='-' else -1 for x in obs_list]
		pg_d[node_num] = (action, obs_list)

	pg_f.close()

	alpha_list = []
	alpha_f = open(alpha_fname, 'r')
	for i,line in enumerate(alpha_f):
		if i%3 == 1:
			line = line.strip().split()
			# print(line)
			weights = np.array(list(map(float, line)))
			alpha_list.append(weights)
	alphas = np.array(alpha_list)
	start_node = np.argmax(alphas.dot(initial_belief))

	return pg_d[start_node][0]
if __name__=="__main__":
    T = [[[0.8,.2],[0.15,0.85]],
         [[0.7,.3],[.10,0.9]]]
    T = np.array(T)
    belief = 0.5
    w = yundi_whittle_exact(T, belief, 140)







