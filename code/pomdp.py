import numpy as np 
import argparse
from itertools import product, combinations
from functools import reduce
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.etree import ElementTree
from xml.dom import minidom
from tqdm import tqdm
import subprocess
import os
import multiprocessing
import platform

# path constants
pomdp_file_folder = 'pomdp_files'
output_folder = 'pg'

# Defaults, probably won't work though.
despot_solver_path = 'pomdpx'
pomdp_solve_path = 'pomdp-solve'

if platform.release() == '3.10.0-957.12.1.el7.x86_64': # cannon
    despot_solver_path = '/n/home10/jkillian/.local/bin/despot'
    pomdp_solve_path = '/n/home10/jkillian/.local/bin/pomdp-solve'

elif platform.release() == '3.10.0-957.12.1.el7.x86_64': # cannon
    despot_solver_path = '/n/home10/jkillian/.local/bin/despot'
    pomdp_solve_path = '/n/home13/amate/projects/tb_bandits/pomdp-solve'

elif platform.release() == '4.15.0-64-generic': # deathstar
    despot_solver_path = '/home/jkillian/.local/bin/despot'
    pomdp_solve_path = '/home/jkillian/.local/bin/pomdp-solve'
    
elif platform.release() == '18.6.0': # Jack Macbook 2019
    despot_solver_path = 'pomdpx'
    pomdp_solve_path = 'pomdp-solve'

elif platform.release() == '18.7.0': # Aditya Macbook 2019
    despot_solver_path = 'pomdpx'
    pomdp_solve_path = 'pomdp-solve'

if not os.path.exists(pomdp_file_folder):
    os.makedirs(pomdp_file_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



def init_pool(o,tc):
    global optimal_results
    global trueCount
    optimal_results = o
    trueCount = tc

def yundi_worker(args):
    """This function will be called by each thread.
    This function can not be a class method.
    """
    # Expand list of args into named args.

    mid, candidate, T, belief, ind, days_remaining, mdp = args
    # del args
    
    if candidate:
        fname, a, obs_space = make_pomdp_given_cost(mid, T, ind, mdp=mdp, discount=1.)
        # print(T)
        initial_belief = np.array([1-belief, belief])
        optimal_action = solve_given_cost(fname, initial_belief, a, days_remaining)
        optimal_results[ind] = optimal_action

        if (optimal_action == 1):
            # Serial-only Portion
            with trueCount.get_lock():
                trueCount.value += 1
        


def yundi_whittle_parallel(N, k, T, belief, days_remaining, mdp=False, n_processes=4, verbose=False):

    # actions=np.zeros(N).astype(int)
    # for (int i = 0; i < numPatients; i++) {
    #     choice[i] = false;
    # }
    
    # boolean[] candidate = new boolean[numPatients];
    actions = np.zeros(N).astype(int)
    candidates = np.ones(N).astype(int)
    # boolean[] optimalResult = new boolean[numPatients];
    # for (int i = 0; i < numPatients; i++) {
    #     candidate[i] = true;
    # }

    remaining = k
    # int left = numCallsPerDay;
    upper = 2
    # upper = patients[0].R[numObservations - 1];
    lower = 0
    # double lower = patients[0].R[0];
    mid = 0
    mids = set()

    worker_args = [[mid, candidates[i], T[i], belief[i], i, days_remaining, mdp] for i in range(N)]
    optimal_results = multiprocessing.Array('i', N)
    
    if verbose:
        print(np.frombuffer(optimal_results.get_obj(),dtype="int32"))

    mid_change = 10000
    mid_change_tol = 1e-6 
    while (remaining > 0) and mid_change > mid_change_tol:
        old_mid = mid
        mid = (upper + lower) / 2.0;
        mid_change = abs(mid - old_mid)
        if verbose:
            print('mid',mid)
        trueCount = multiprocessing.Value('i', 0)
        
        pool = None
        if n_processes > 0:
            pool = multiprocessing.Pool(processes=n_processes, initializer=init_pool, initargs=(optimal_results, trueCount))
        else:
            init_pool(optimal_results, trueCount)

        for i in range(N):
            worker_args[i][0] = mid
            worker_args[i][1] = candidates[i]

        # print('he')
        # map(yundi_worker, worker_args)
        try:
            # Spawn up to 9999999 jobs, I think this is the maximum possible.
            # I do not know what happens if you exceed this.
            if n_processes > 0:
                pool.map_async(yundi_worker, worker_args).get(9999999)
            elif n_processes <= 0:
                list(map(yundi_worker, worker_args))

        except KeyboardInterrupt:
            # Allow ^C to interrupt from any thread.
            sys.stdout.write('\033[0m')
            sys.stdout.write('User Interupt\n')
        if pool:
            pool.close()

        if (trueCount.value <= remaining):
            for i in range(N):
                if (candidates[i] == 1 and optimal_results[i] == 1):
                    candidates[i] = 0;
                    actions[i] = 1;
                
            upper = mid;
            remaining -= trueCount.value;

        else:
            for i in range(N):
                if candidates[i] == 1 and optimal_results[i] == 0:
                    candidates[i] = 0;
                
            
            lower = mid;

    # actions = np.frombuffer(actions.get_obj(),dtype="int32")
    if (remaining > 0):
        if verbose:
            print("breaking tie randomly")
        while(actions.sum() < k):
            print([i for i in range(candidates.shape[0]) if candidates[i]==1])
            ind = np.random.choice([i for i in range(candidates.shape[0]) if candidates[i]==1])
            actions[ind] = 1

    if actions.sum() != k:
        raise ValueError('wrong number of actions!',actions.sum())

    #print('actions:', ' '.join(list(map(str, actions))))
    mids.add(mid)
    return actions




def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def make_char(x):
    return chr(x+65)

def tup_print(tups, joiner='', f=None):
    tup_strs = [joiner.join(list(map(make_char, tup))) for tup in tups]
    # print(' '.join(tup_strs),file=f)
    return tup_strs


def print_POMDP(combined_action_space, combined_state_space, combined_observation_space,
        T_matrices, O_matrices, R,
         pomdp_filename, discount, root):

    # fname = os.path.join(root,pomdp_file_folder)
    # fname = os.path.join(fname, pomdp_filename)
    fname = os.path.join(pomdp_file_folder, pomdp_filename)
    fout = open(fname,'w')

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

    for i, action in tqdm(enumerate(action_space_strs)):
        print('T:%s'%action,file=fout)
        for row in T_matrices[i]:
            print(' '.join(list(map(str, row))), file=fout)
        print(file=fout)

    for i, action in tqdm(enumerate(action_space_strs)):
        print('O:%s'%action,file=fout)
        for row in O_matrices[i]:
            print(' '.join(list(map(str, row))), file=fout)
        print(file=fout)

    for i,state in tqdm(enumerate(state_space_strs)):
        print('R:* : * : %s : * %i' % (state, R[i]), file=fout)

    fout.close()


def print_pomdpx(combined_action_space, combined_state_space, combined_observation_space,
        T_matrices, O_matrices, R,
         pomdp_filename, discount, file_root):

    root = Element('pomdpx', version='0.1', id='tagxmlfac')
    root.set('xmlns:xsi','http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation','pomdpx.xsd')

    description = SubElement(root, 'Description')
    description.text='Patient Adherence POMDP'

    discount_xml = SubElement(root, 'Discount')
    discount_xml.text = str(discount)

    variable = SubElement(root, 'Variable')
    statevar = SubElement(variable, 'StateVar', vnamePrev='s', vnameCurr='s_prime', fullyObs='false')
    valenum = SubElement(statevar, 'ValueEnum')
    state_space_strs = tup_print(combined_state_space, joiner='s')
    valenum.text = ' '.join(state_space_strs)

    obsvar = SubElement(variable, 'ObsVar', vname='o')
    valenum = SubElement(obsvar, 'ValueEnum')
    obs_space_strs = tup_print(combined_observation_space, joiner='o')
    valenum.text = ' '.join(obs_space_strs)

    actionvar = SubElement(variable, 'ActionVar', vname='a')
    valenum = SubElement(actionvar, 'ValueEnum')
    action_space_strs = tup_print(combined_action_space, joiner='a')
    valenum.text = ' '.join(action_space_strs)

    reward_var = SubElement(variable, 'RewardVar', vname='r')


    # b
    init_belief = SubElement(root, 'InitialStateBelief')
    cond_prob = SubElement(init_belief, 'CondProb')
    var = SubElement(cond_prob, 'Var')
    var.text = 's'
    parent = SubElement(cond_prob, 'Parent')
    parent.text = 'null'

    param = SubElement(cond_prob, 'Parameter', type='TBL')

    entry_xml = SubElement(param, 'Entry')
    instance = SubElement(entry_xml, 'Instance')
    instance.text = '-'
    prob_table = SubElement(entry_xml, 'ProbTable')
    prob_table.text = 'uniform'


    # T 
    state_transition_function = SubElement(root, 'StateTransitionFunction')
    cond_prob = SubElement(state_transition_function, 'CondProb')
    var = SubElement(cond_prob, 'Var')
    var.text = 's_prime'
    parent = SubElement(cond_prob, 'Parent')
    parent.text = 'a s'

    param = SubElement(cond_prob, 'Parameter', type='TBL')

    print('T')
    for i, action in tqdm(enumerate(action_space_strs)):
        for j, row in enumerate(T_matrices[i]):
            for k, entry in enumerate(row):
                entry_xml = SubElement(param, 'Entry')
                instance = SubElement(entry_xml, 'Instance')
                instance.text = ' '.join( [action, state_space_strs[j], state_space_strs[k] ]  )
                prob_table = SubElement(entry_xml, 'ProbTable')
                prob_table.text = str(entry)


    # O
    print('O')
    obs_function = SubElement(root, 'ObsFunction')
    cond_prob = SubElement(obs_function, 'CondProb')
    var = SubElement(cond_prob, 'Var')
    var.text = 'o'
    parent = SubElement(cond_prob, 'Parent')
    parent.text = 'a s_prime'

    param = SubElement(cond_prob, 'Parameter', type='TBL')

    for i, action in tqdm(enumerate(action_space_strs)):
        for j, row in enumerate(O_matrices[i]):
            for k, entry in enumerate(row):
                entry_xml = SubElement(param, 'Entry')
                instance = SubElement(entry_xml, 'Instance')
                instance.text = ' '.join( [action, state_space_strs[j], obs_space_strs[k] ]  )
                prob_table = SubElement(entry_xml, 'ProbTable')
                prob_table.text = str(entry)



    # R
    print('R')
    reward_function = SubElement(root, 'RewardFunction')
    func = SubElement(reward_function, 'Func')
    var = SubElement(func, 'Var')
    var.text = 'r'
    parent = SubElement(func, 'Parent')
    parent.text = 'a s_prime' # MAYBE CHANGE THIS IF DOESN'T WORK

    param = SubElement(func, 'Parameter', type='TBL')

    for i, state_space_str in tqdm(enumerate(state_space_strs)):
        entry_xml = SubElement(param, 'Entry')
        instance = SubElement(entry_xml, 'Instance')
        instance.text = ' '.join( ['*', state_space_str]  )
        value_table = SubElement(entry_xml, 'ValueTable')
        value_table.text = str(R[i])

    # for i,state in enumerate(state_space_strs):
    #     print('R:* : * : %s : * %i' % (state, R[i]), file=fout)

    # fname = os.path.join(file_root,pomdp_file_folder)
    fname = os.path.join(pomdp_file_folder, pomdp_filename)
    fout = open(fname, 'w')
    print(prettify(root), file=fout)
    fout.close()


# Create Combined Patient Adherence POMDP
# n = number of patients
# k = patients per day one can call
# T = n-length vector of transition matrices, which are each num_actions x num_states x num_states
# O = n-length vector of observation matrices, which are each num_actions x num_states x num_observations=3
# dicsount = discount in infinite horizon case
# method = {'exact', 'DESPOT'}

def make_pomdp(n, k, T, O, discount=0.95, method='exact', root='.'):

    num_states = 2
    num_observations = 3


    states_per_patient = np.arange(num_states)
    combined_state_space = list(product(states_per_patient, repeat=n))
    # print('State Space')
    # print(combined_state_space)
    # print()

    patient_indices = np.arange(n)
    combined_action_space = list(combinations(patient_indices, k))
    # print('Action Space')
    # print(combined_action_space)
    # print()

    observations_per_patient = np.arange(num_observations)
    combined_observation_space = list(product(observations_per_patient, repeat=n))
    # print('Observation Space')
    # print(combined_observation_space)
    # print()

    # 0 is not adhering, 1 is adhering
    # 0 is no action, 1 is action
    # T0 = np.array([
    #     [0.9, 0.1],
    #     [0.1, 0.9]
    # ])
    # T1 = np.array([
    #     [0.6, 0.4],
    #     [0.01, 0.99]
    # ])

    # d = {
    #     0: T0,
    #     1: T1
    # }

    T_matrices = []
    for a in tqdm(combined_action_space):
        one_hot_patients = np.zeros(n).astype(int)
        one_hot_patients[list(a)] = 1

        inputs = [ T[ind][one_hot] for ind, one_hot in enumerate(one_hot_patients) ]

        mat = reduce(lambda a,b : np.kron(a,b),inputs)
        T_matrices.append(mat)


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

    O_matrices = []
    for a in tqdm(combined_action_space):
        one_hot_patients = np.zeros(n)
        one_hot_patients[list(a)] = 1

        # all patients have same O for now
        inputs = [d[x] for x in one_hot_patients]
        # print(inputs)
        mat = reduce(lambda a,b : np.kron(a,b),inputs)
        # print(mat)
        O_matrices.append(mat)




    # R: <action> : <start-state> : <end-state> : <observation> %f
    # so do 
    # and compute for all end states such that it is the sum of the 1's in the end state
    # R: * : * : <end-state> : * %f
    R = [sum(x) for x in tqdm(combined_state_space)]

    if method =='exact':
        pomdp_filename = 'combined_patient_p=%i_k=%i.POMDP' %(n, k)
        print_POMDP(combined_action_space, combined_state_space, combined_observation_space,
            T_matrices, O_matrices, R, pomdp_filename = pomdp_filename, 
            discount=discount, root=root)
        return pomdp_filename, combined_action_space, combined_observation_space
        
    elif method=='DESPOT':
        pomdp_filename = 'combined_patient_p=%i_k=%i.pomdpx' %(n, k)
        print_pomdpx(combined_action_space, combined_state_space, combined_observation_space,
            T_matrices, O_matrices, R, pomdp_filename = pomdp_filename, 
            discount=discount, file_root=root)

        return pomdp_filename

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


def make_pomdp_given_cost(C, T, ind, mdp=False, discount=0.95):

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
    # print('Observation Space')
    # print(combined_observation_space)
    # print()

    # 0 is not adhering, 1 is adhering
    # 0 is no action, 1 is action
    # T0 = np.array([
    #     [0.9, 0.1],
    #     [0.1, 0.9]
    # ])
    # T1 = np.array([
    #     [0.6, 0.4],
    #     [0.01, 0.99]
    # ])

    # d = {
    #     0: T0,
    #     1: T1
    # }

    T_matrices = T # first one should be no action, second is action


    O0 = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ]
    if mdp:
        O0 = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
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


def solve_given_cost(fname, initial_belief, action_space, days_remaining):

    outname = os.path.join(output_folder, fname)
    # outname = os.path.join(root, outname)

    fname = os.path.join(pomdp_file_folder, fname)
    # fname = os.path.join(root, fname)

    subprocess.check_output([pomdp_solve_path, '-pomdp', fname, '-o', outname, '-horizon', str(days_remaining)])
    # subprocess.run([pomdp_solve_path, '-pomdp', fname, '-o', outname, '-horizon', str(days_remaining)])

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
    # print(alphas)

    start_node = np.argmax(alphas.dot(initial_belief))
    # print(start_node)

    return pg_d[start_node][0]


def solve(fname, initial_belief, action_space, root='.'):


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
        patients_getting_action = action_space[action]
        n = np.array(action_space).reshape(-1).max()+1
        one_hot_patients = np.zeros(n)
        one_hot_patients[list(patients_getting_action)] = 1
        action = one_hot_patients
        
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
    alpha_f.close()
    # print(alphas)

    start_node = np.argmax(alphas.dot(initial_belief))
    # print(start_node)

    return start_node, pg_d


def despot(fname, policy_depth=180, timeout=0.01, root='.'):
    
    fname = os.path.join(pomdp_file_folder, fname)
    # fname = os.path.join(root, fname)

    a = subprocess.check_output([despot_solver_path, '-m', fname, 
        '--timeout', str(timeout), '--simlen', str(policy_depth),
        '--silence'])

    a = a.decode("utf-8").strip().split('\n')
    result = a[-2] # Average total undiscounted reward (stderr) = 117 (0)
    result = result.split(' = ')[-1]
    result = result.split(' (')[0]
    result = float(result)
    print(result)
    print('done!')
    return result



