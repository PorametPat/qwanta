# %%

from qwanta import Xperiment
import numpy as np
import time
import ray
import sys
import dill
import pandas as pd

ray.init()

@ray.remote
def execute(parameter):

    index = parameter['index']
    loss = parameter['loss rate'] # dB/km
    depo_prob = parameter['depolarizing rate']
    gate_error = parameter['gate error rate']
    measurement_error = parameter['measurement error']
    memory_time = parameter['memory error']
    repeat_th = parameter['trajectory']

    num_hops = parameter['number of hops']
    num_nodes = num_hops + 1

    node_info = {f'Node {i}': {'coordinate': (int(i*100), 0, 0)} for i in range(num_nodes)}
    edge_info = {
        (f'Node {i}', f'Node {i+1}'): {
        'connection-type': 'Space',
        'depolarlizing error': [1 - depo_prob, depo_prob/3, depo_prob/3, depo_prob/3],
        'loss': loss,
        'light speed': 300000,
        'Pulse rate': 0.0001,
        f'Node {i}':{
            'gate error': gate_error,
            'measurement error': measurement_error,
            'memory function': memory_time
        },
        f'Node {i+1}':{
            'gate error': gate_error,
            'measurement error': measurement_error,
            'memory function': memory_time
        },
        }
    for i in range(num_hops)}

    exps = Xperiment(
        timelines_path = f'exper_id2_selectedStats_{num_hops}hops.xlsx',
        nodes_info_exp = node_info,
        edges_info_exp = edge_info,
        gate_error = gate_error,
        measurement_error = measurement_error,
        memory_time = memory_time,
        experiment = f'exp_id2_StrategiesAnalysisNoiseSensitivity_p{index}_r{repeat_th}'
    )

    exps.execute(save_result=True)

    return True

# %%
loss_list =  np.array([0.03])
p_dep_list = np.array([0.025])
gate_error_list = np.array([0, 0.0025, 0.005, 0.0075, 0.01])
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
    print(f'Finish {hops}.')

start_parameter_index = 0  if len(sys.argv) == 1 else int(sys.argv[1])

start_time = time.time()
results = ray.get([execute.remote(i) for i in parameters_set[start_parameter_index:]])
print('Simulated Time: ', time.time() - start_time)

# %%
message_log = 'exp_id2_StrategiesAnalysisNoiseSensitivity'

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
                                    'base Bell pairs': exp['Base Resources Produced']
                                }
                                parameters_set.append(data)
                        index += 1

DataFrame = pd.DataFrame(parameters_set)
DataFrame.to_csv('exp_id2_StrategiesAnalysisNoiseSensitivity_Extracted_Data.csv', index=False)

