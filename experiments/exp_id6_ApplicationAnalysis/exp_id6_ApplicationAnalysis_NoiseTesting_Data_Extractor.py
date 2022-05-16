from qwanta import Xperiment
import numpy as np
import time
import ray
import sys
import dill
import pandas as pd

loss_list =  np.array([0, 0.3])
p_dep_list = np.array([0, 0.025])
gate_error_list = np.array([0, 0.001])
mem_error_list = np.array([np.inf, 0.01])
measurement_error_list =  np.array([0, 0.01])
number_of_hops_list = np.array([2, 4, 8]) 
num_trajectories = 10
exp_names = ['0G', 'E2E-1G-Ss-Dp', '1G-Ss-Dp', '1-2G-DirectedEncoded', 'HG-END2ENDPurifiedEncoded', '2G-NonLocalCNOT']
message_log = 'exp_id6_ApplicationAnalysis_NoiseTesting_modified_result_extraction'
distance = 100

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

data_set = []

for params_set in parameters_set:
    for exp_name in exp_names:
        
        index = params_set['index']
        loss = params_set['loss rate']
        p_dep = params_set['depolarizing rate']
        gate_error = params_set['gate error rate']
        mem_error = params_set['memory error']
        measure_error = params_set['measurement error']
        hops = params_set['number of hops']
        trajectory = params_set['trajectory']

        # Read file 
        with open(f"result/Result_{message_log}_p{index}_r{trajectory}_{exp_name}.pkl", "rb") as f:
            exp = dill.load(f)

        Node_left = exp['throughtputEdges'][0]
        Node_right = exp['throughtputEdges'][1]

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
            'fidelity': exp['fidelity'],
            'total time': exp['Time used'],
            'fidelity estimation time': exp['Fidelity Estimation Time'],
            'fidelity estimated edges': f"{Node_left}, {Node_right}",
            'label resource produced': exp['Resources Produced'][f'{Node_left}-{Node_right}']['k'],
            'base Bell pairs attempted': exp['Base Resources Attempt'],
            'distance': distance,
            'XX commute': exp['Commutation inforamtion']['XX']['commute'],
            'XX anti-commute': exp['Commutation inforamtion']['XX']['anti-commute'],
            'YY commute': exp['Commutation inforamtion']['YY']['commute'],
            'YY anti-commute': exp['Commutation inforamtion']['YY']['anti-commute'],
            'ZZ commute': exp['Commutation inforamtion']['ZZ']['commute'],
            'ZZ anti-commute': exp['Commutation inforamtion']['ZZ']['anti-commute'],
        }
        data_set.append(data)

DataFrame = pd.DataFrame(data_set)
DataFrame.to_csv('exp_id6_ApplicationAnalysis_NoiseTesting_Extracted_Data_modified.csv', index=False)
