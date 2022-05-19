from qwanta import Xperiment
import numpy as np
import time
import ray
import sys
import dill
import pandas as pd

# %%
loss_list =  np.array([0.03])
p_dep_list = np.array([0.025])
gate_error_list = np.array([0, 0.0005, 0.001, 0.0015, 0.002])
mem_error_list = np.array([0.01])
measurement_error_list =  np.array([0, 0.0025, 0.005, 0.0075, 0.01])
number_of_hops_list = np.array([2, 4, 8, 16]) 
num_trajectories = 10
exp_names = ['0G', '1G-Ss-Dp', '2G-NonLocalCNOT', '1-2G-DirectedEncoded', 
             'HG-END2ENDPurifiedEncoded', '2G-NonLocalCNOT-Perfect', 
             '1-2G-DirectedEncoded-Perfect']

parameters_set = []; index = 0
for hops in number_of_hops_list:
    for loss in loss_list:
        for p_dep in p_dep_list:
            for gate_error in gate_error_list:
                for mem_error in mem_error_list:
                    for measure_error in measurement_error_list:
                        for trajectory in range(num_trajectories):
                            parameters_set.append({
                                'index': index,
                                'loss rate': loss, 
                                'depolarizing rate': p_dep, 
                                'gate error rate': gate_error, 
                                'memory error': mem_error,
                                'measurement error': measure_error,
                                'number of hops': hops,
                                'trajectory': trajectory
                                })
                        index += 1

message_log = 'exp_id2_StrategiesAnalysisOverviewResults_fixed'

parameters_set = []; index = 0
for hops in number_of_hops_list:
    for loss in loss_list:
        for p_dep in p_dep_list:
            for gate_error in gate_error_list:
                for mem_error in mem_error_list:
                    for measure_error in measurement_error_list:
                        for exp_name in exp_names:
                            for trajectory in range(num_trajectories):
                                
                                # Read file 
                                with open(f"result/Result_{message_log}_p{index}_r{trajectory}_{exp_name}.pkl", "rb") as f:
                                    exp = dill.load(f)

                                data = {
                                    'index': index,
                                    'loss rate': loss, 
                                    'depolarizing rate': p_dep, 
                                    'gate error rate': gate_error, 
                                    'memory error': mem_error,
                                    'measurement error': measure_error,
                                    'number of hops': hops,
                                    'trajectory': trajectory,
                                    'experiment': exp_name,
                                    'number of hops': hops,
                                    'fidelity': exp['fidelity'],
                                    'total time': exp['Time used'],
                                }
                                parameters_set.append(data)
                        index += 1

DataFrame = pd.DataFrame(parameters_set)
DataFrame.to_csv('exp_id2_StrategiesAnalysisOverviewResults_fixed_Extracted_Data.csv', index=False)

