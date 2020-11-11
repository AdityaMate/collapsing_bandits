# Collapsing Bandits
Code repo for "Collapsing Bandits" (NeurIPS 2020) simulation environment

Example Usage: python3 adherence_simulation.py -N 10 -n 50 -k 5 -s 0 -ws 0 -ls 0 -l 180 -d simulated
- N: Number of trials to average over
- n: Number of processes 
- k: Number of active actions allowed each day
- s, ws, ls: Seeds for random generator streams.
- L: Length of simulation (default is 180 rounds, to simulate 6 months of TB adherence data)
- d: Dataset to use (real/simulated). Note that real TB data is not publicly available, so has been removed from codebase, so the code can only run in the "-d simulated" mode.
 
### Note: The above command will run the simulation for the faster policies only since the baselines take lot of time to run. To run other baselines add a -pc flag to run a specific policy, with parameters as exlplained below: 

For a specific policy python3 adherence_simulation.py -N 10 -n 50 -k 5 -s 0 -ws 0 -ls 0 -l 180 -pc 5 
- -pc 0: No actions 
- -pc 1: All actions are active
- -pc 2: Random actions
- -pc 3: Myopic policy 
- -pc 5: Threshold Whittle
- -pc 10: Qian et al.
- -pc 14: Oracle 
