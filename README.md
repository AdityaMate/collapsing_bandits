# Collapsing Bandits
Code repo for "Collapsing Bandits" (NeurIPS 2020) simulation environment. Main paper is [here](https://papers.nips.cc/paper/2020/file/b460cf6b09878b00a3e1ad4c72344ccd-Paper.pdf) and full paper with appendix is [here](https://teamcore.seas.harvard.edu/files/teamcore/files/collapsing_bandits_full_paper_camready.pdf): 
```
@article{mate2020collapsing,
  title={Collapsing Bandits and Their Application to Public Health Intervention},
  author={Mate, Aditya and Killian, Jackson and Xu, Haifeng and Perrault, Andrew and Tambe, Milind},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
## Execution
Example Usage: python3 adherence_simulation.py -N 10 -n 50 -k 5 -s 0 -ws 0 -ls 0 -l 180 -d simulated
- `- N`: Number of trials to average over
- `- n`: Number of processes 
- `- k`: Number of active actions allowed each day
- `- s, ws, ls`: Seeds for random generator streams.
- `- L`: Length of simulation (default is 180 rounds, to simulate 6 months of TB adherence data)
- `- d`: Dataset to use (real/simulated). Note that real TB data is not publicly available, so has been removed from codebase, so the code can only run in the "-d simulated" mode.
 
### Note: The above command will run the simulation for the faster policies only since the baselines take lot of time to run. To run other baselines add a -pc flag to run a specific policy, with parameters as exlplained below: 

For a specific policy python3 adherence_simulation.py -N 10 -n 50 -k 5 -s 0 -ws 0 -ls 0 -l 180 -pc 5 
- `-pc 0`: No actions 
- `-pc 1`: All actions are active
- `-pc 2`: Random actions
- `-pc 3`: Myopic policy 
- `-pc 5`: Threshold Whittle
- `-pc 10`: Qian et al.
- `-pc 14`: Oracle 
