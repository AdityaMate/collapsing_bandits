# POLICY NAMES ***
# 0: never call    1: Call all patients everyday     2: Randomly pick k patients to call
# 3: Myopic policy    4: pomdp  5: whittle    6: 2-day look ahead    7:oracle
# 8: oracle_whittle   9: despot 10: yundi's whittle index 11,12,13: Lily 
# 14: MDP oracle     15: round robin  16: new new whittle(fast)  17: fast whittle 18: Buggy whittle

NUM_TRIALS=20
TREATMENT_LENGTH=180
# KFRAC=25
cdir=$(pwd)
badf=0.2
datatype='real'
t1f=0
# learning_mode=3
SAVENAME="final_thompson"
pc=17

for (( trial_i = 0; trial_i < $NUM_TRIALS; trial_i++))
do
        for n in 100
        do
                for KFRAC in 5 10 25 # fast whittle
                do
                    # Run whittle, random, and call everyone without learning
                    # python3 $cdir/adherence_simulation.py -n $n -kp $KFRAC -l $TREATMENT_LENGTH -f $cdir -N $NUM_TRIALS -d $datatype -s 0 -ws 0 -ls 0 -sv 'thompson_sampling'  -badf $badf -tr $trial_i --learning_option 0 -pc $pc
                    # python3 $cdir/adherence_simulation.py -n $n -kp $KFRAC -l $TREATMENT_LENGTH -f $cdir -N $NUM_TRIALS -d $datatype -s 0 -ws 0 -ls 0 -sv 'thompson_sampling'  -badf $badf -tr $trial_i --learning_option 0 -pc 1
                    # python3 $cdir/adherence_simulation.py -n $n -kp $KFRAC -l $TREATMENT_LENGTH -f $cdir -N $NUM_TRIALS -d $datatype -s 0 -ws 0 -ls 0 -sv 'thompson_sampling'  -badf $badf -tr $trial_i --learning_option 0 -pc 2
                    sbatch single_point_adherence_thompson_evaluation.sh $n $KFRAC  $TREATMENT_LENGTH $NUM_TRIALS $datatype 0 0 0 $SAVENAME $badf $trial_i 0 0 0 0
                    # sbatch single_point_adherence_thompson_evaluation.sh $n $KFRAC  $TREATMENT_LENGTH $NUM_TRIALS $datatype 0 0 0 $SAVENAME $badf $trial_i 0 1 0 0
                    # sbatch single_point_adherence_thompson_evaluation.sh $n $KFRAC  $TREATMENT_LENGTH $NUM_TRIALS $datatype 0 0 0 $SAVENAME $badf $trial_i 0 2 0 0
                    # sbatch single_point_adherence_thompson_evaluation.sh $n $KFRAC  $TREATMENT_LENGTH $NUM_TRIALS $datatype 0 0 0 $SAVENAME $badf $trial_i 0 $pc 0 0
                    
            #         for learning_mode in 1 3
            #         do
            #             # sbatch single_point_adherence_thompson_evaluation.sh $n $KFRAC  $TREATMENT_LENGTH $NUM_TRIALS $datatype 0 0 0 $SAVENAME $badf $trial_i $learning_mode 14 0 0
            #           	#sh trial.sh $n 10 180 1 'real' 0  $policy trialN20k10RealL180 $trial_i
        				# # for BUFFER_LENGTH in 0 1 2 4 7 14
            #             for BUFFER_LENGTH in 0 1 4 7
        				# do
            #                 # python3 $cdir/adherence_simulation.py -n $n -kp $KFRAC -l $TREATMENT_LENGTH -f $cdir -N $NUM_TRIALS -d $datatype -s 0 -ws 0 -ls 0 -sv 'thompson_sampling'  -badf $badf -tr $trial_i --learning_option 1 -pc $pc -bl $BUFFER_LENGTH -t1f $t1f
	           #              # sbatch single_point_adherence_thompson_evaluation.sh $n $KFRAC  $TREATMENT_LENGTH $NUM_TRIALS $datatype 0 0 0 $SAVENAME $badf $trial_i $learning_mode $pc $BUFFER_LENGTH $t1f
            #             done
            #         done
                done
        done
done
