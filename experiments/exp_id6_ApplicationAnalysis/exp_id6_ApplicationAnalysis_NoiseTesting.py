from qwanta import Xperiment
import numpy as np
import time
import ray
import sys
import dill
import pandas as pd

ray.init()

@ray.remote
def execute(parameter, distance, message_log):

    index = parameter['index']
    loss = parameter['loss rate'] # dB/km
    depo_prob = parameter['depolarizing rate']
    gate_error = parameter['gate error rate']
    measurement_error = parameter['measurement error']
    memory_time = parameter['memory error']
    repeat_th = parameter['trajectory']

    num_hops = parameter['number of hops']
    num_nodes = num_hops + 1

    node_info = {f'Node {i}': {'coordinate': (dis, 0, 0)} for i, dis in enumerate(np.linspace(0, distance, num_nodes))}
    edge_info = {
        (f'Node {i}', f'Node {i+1}'): {
        'connection-type': 'Space',
        'depolarlizing error': [1 - depo_prob, depo_prob/3, depo_prob/3, depo_prob/3],
        'loss': loss,
        'light speed': 300000,
        'Pulse rate': 0.00001,
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
        timelines_path = f'exper_id6_selectedStats_{num_hops}hops.xlsx',
        nodes_info_exp = node_info,
        edges_info_exp = edge_info,
        gate_error = gate_error,
        measurement_error = measurement_error,
        memory_time = memory_time,
        experiment = f'{message_log}_p{index}_r{repeat_th}'
    )

    if exps.validate():
        exps.execute(save_result=True)
    else:
        raise ValueError('Validation failed')

    return True

loss_list =  np.array([0, 0.3])
p_dep_list = np.array([0, 0.025])
gate_error_list = np.array([0, 0.001])
mem_error_list = np.array([np.inf, 0.01])
measurement_error_list =  np.array([0, 0.01])
number_of_hops_list = np.array([2, 4, 8]) 
num_trajectories = 10
exp_names = ['0G', 'E2E-1G-Ss-Dp', '1G-Ss-Dp', '1-2G-DirectedEncoded', 'HG-END2ENDPurifiedEncoded', '2G-NonLocalCNOT']
message_log = 'exp_id6_ApplicationAnalysis_NoiseTesting'
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

start_parameter_index = 0  if len(sys.argv) == 1 else int(sys.argv[1])

start_time = time.time()
results = ray.get([execute.remote(i, distance, message_log) for i in parameters_set[start_parameter_index:]])
print('Simulated Time: ', time.time() - start_time)

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
                                    'distance': distance
                                }
                                parameters_set.append(data)
                        index += 1

DataFrame = pd.DataFrame(parameters_set)
DataFrame.to_csv('exp_id6_ApplicationAnalysis_NoiseTesting_Extracted_Data.csv', index=False)

