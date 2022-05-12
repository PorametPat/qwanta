import simpy
import networkx as nx
from .Qubit.qubit import PhysicalQubit
import os 
import pandas as pd
import seaborn as sns
import dill
import ast
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from pyvis.network import Network
import random
import numpy as np

from .QuantumProcess import _EntanglementPurification, _EntanglementSwapping, _GenerateLogicalResource, _GeneratePhyscialResource, _VirtualStateTomography
from .SubProcess import _TimeLag


class Xperiment:

    def __init__(self, timelines_path, 
                       nodes_info_exp, 
                       edges_info_exp, 
                       gate_error=0,
                       measurement_error=0,
                       memory_time=1,
                       strategies_list=None, 
                       resources_dict=None,
                       experiment=None,
                       sim_time=None,
                       label_record='Physical'
                ):

        if isinstance(strategies_list, str):
            strategies_list = [strategies_list]

        timelines = pd.read_excel(timelines_path, sheet_name=strategies_list)

        # Convert Edges and Num trials to legit type
        for exp in timelines:
            timelines[exp]['Edges'] = timelines[exp]['Edges'].transform(lambda x: ast.literal_eval(x))
            timelines[exp]['Num Trials'] = timelines[exp]['Num Trials'].transform(lambda x: True if x == 'True' else x)
            timelines[exp]['Num Trials'] = timelines[exp]['Num Trials'].transform(lambda x: int(x) if int(x) > 1 else x) 
            timelines[exp] = timelines[exp].to_dict('records')

        self.timelines = timelines
        self.strategies_list = list(self.timelines.keys())

        # Memory error
        if isinstance(memory_time, float) or isinstance(memory_time, int):
            def memory_error_function(time, tau=memory_time):
                p = 3*(np.e**(-1*(time/tau)))/4 + 0.25
                return [p, (1- p)/3, (1- p)/3, (1- p)/3]
            memory_function = memory_error_function
        else:
            memory_function = memory_time

        default_resources_dict = {
            'numPhysicalBuffer': 20,
            'numInternalEncodingBuffer': 20,
            'numInternalDetectingBuffer': 10,
            'numInternalInterfaceBuffer': 2
        }
        if resources_dict is None:
            valid_resources_dict = default_resources_dict
        else:
            valid_resources_dict = resources_dict if all([ num in resources_dict for num in list(default_resources_dict.keys())]) else default_resources_dict

        for exp in nodes_info_exp:
            nodes_info_exp = {**valid_resources_dict, **nodes_info_exp}

        self.nodes_info_exp = { exp: nodes_info_exp for exp in self.strategies_list }
        self.edges_info_exp = { exp: edges_info_exp for exp in self.strategies_list }

        self.gate_errors = { exp: gate_error for exp in self.strategies_list }
        self.measurement_errors = { exp: measurement_error for exp in self.strategies_list }
        self.memory_functions = { exp: memory_function for exp in self.strategies_list }

        self.sim_times = { exp: sim_time for exp in self.strategies_list}
        self.label_records = { exp: label_record for exp in self.strategies_list}
        
        self.configurations = {
            exp : Configuration(
                topology = self.edges_info_exp[exp], 
                timeline = self.timelines[exp], 
                nodes_info = self.nodes_info_exp[exp], 
                memFunc = self.memory_functions[exp], 
                gate_error = self.gate_errors[exp], 
                measurementError = self.measurement_errors[exp],
                experiment = experiment, # Record experiment set with experiment name
                message = exp,
                sim_time = self.sim_times[exp],
                label_record = self.label_records[exp]
            )
        for exp in self.strategies_list }

        self.QuantumNetworks = {
            exp : QuantumNetwork(self.configurations[exp])
        for exp in self.strategies_list }

        self.graphs = {
            exp : self.configurations[exp].NetworkTopology
        for exp in self.strategies_list}
        
        self.process_graph = {}

    def validate(self, vis=False, get_table=False):
        validate_table = []

        exper_allGreen = 0
        for exper in self.strategies_list:

            print(f'\nValidating experiment: {exper}')
            print('---------------------------------------------')

            topology =self.edges_info_exp[exper]
            timeline = self.timelines[exper]
            sim_time = self.sim_times[exper]
            gate_error = self.gate_errors[exper]
            measurement_error = self.measurement_errors[exper]
            memory_error =self.memory_functions[exper]
            nodes_info = {}
            for node in self.nodes_info_exp[exper]:
                if node not in ['numPhysicalBuffer', 'numInternalEncodingBuffer', 'numInternalDetectingBuffer', 'numInternalInterfaceBuffer']:
                    nodes_info[node] = self.nodes_info_exp[exper][node]

            # Detect dynamic nodes
            Dynamic_flag = False
            for node in nodes_info:
                if callable(nodes_info[node]['coordinate']):
                    Dynamic_flag = True

            # Validate parameter
            Parameters_test = {
                'loss': True,
                'depolarizing error': True,
                'gate error': True,
                'memory error': True,
                'measurement error': True,
                'Network': 'Dynamic' if Dynamic_flag else 'Stationary'
            }

            # loss and depolarizing_error
            for edge in topology:
                loss = topology[edge]['loss'] # dB/km
                if type(loss) is not float and type(loss) is not int:
                    Parameters_test['loss'] = False

                p_dep = topology[edge]['depolarlizing error']
                if type(p_dep) is list:
                    if len(p_dep) != 4:
                        print(f'[{exper}] length depolarizing probability of {edge} is not 4')
                        Parameters_test['depolarizing error'] = False
                    if int(round(sum(np.abs(p_dep)), 4)) != 1:
                        print(f'[{exper}] WARNING sum of depolarizing probability of {edge} is not 1 to decimal 4.')
                        Parameters_test['depolarizing error'] = False

            # gate_error
            if gate_error > 1 or gate_error < 0:
                print(f'[{exper}] WARNING gate error probability is not a valid value.')
                Parameters_test['gate error'] = False

            # memory_error
            if callable(memory_error):
                for i in range(10):
                    x = random.random()
                    p = memory_error(x)
                    if len(p) != 4:
                        print(f'[{exper}] length of memory error is not 4')
                        Parameters_test['memory error'] = False
                    if int(round(sum(p), 2)) != 1:
                        print(f'[{exper}] WARNING sum of memory error probability of {edge} is not 1 to decimal 2.')
            else:
                print(f'[{exper}] WARNING memory error is not callable function')
                if len(memory_error) != 4:
                    print(f'[{exper}] length of memory error is not 4')
                    Parameters_test['memory error'] = False
            
            # measurement_error
            if measurement_error > 1:
                print(f'[{exper}] measurement error provided is more than 1')
                Parameters_test['measurement error'] = False

            # Valiate Process-flow and number of resource

            G = nx.Graph()
            for edge in topology:
                G.add_edge(edge[0], edge[1]) 
            nx.set_edge_attributes(G, topology)

            print('Limited Process: ')
            limited_processes = []
            for process in timeline:
                if type(process['Num Trials'] )is int:
                    limited_processes.append(process['Main Process'])

                    if tuple(process['Edges']) == (list(G.nodes())[0], list(G.nodes())[-1]):
                        is_end_to_end = f'End-to-end process -> {process["Edges"]}'
                    else:
                        is_end_to_end = f'Non-E2E process -> {process["Edges"]}'

                    print(f'{len(limited_processes)}. ' ,process['Main Process'], f': {is_end_to_end}')
            

            if len(limited_processes) == 0:
            # Detect no limited process simulation
                print(f'[{exper}] experiment has no limited process...')

                if sim_time is None:
                    print(f'[{exper}] WARNING: simulation might not terminate')
            
            if sim_time is not None:
                print(f'[{exper}] experiment will end with simulation time limit: {sim_time}s')

            if Dynamic_flag:
                print(f'[{exper}] experiment contains dynamic nodes...')
                if sim_time is None:
                    print(f'[{exper}] experiment will terminate only if limited process is finish, please make sure that loss and distance provided are reasonable as simulation might take insanely much time to finish...')

            if nx.is_connected(G):
                print('Topology provied is connected graph...')
            else:
                print('The topology provided is not connected, this might not be what is expected...')

            # Validate if all edges have generate physical resource

            num_gpr = 0
            missing_process_edge = []
            for edge in G.edges:
                
                for process in timeline:
                    if process['Main Process'] in ['Generate physical Bell pair']:
                        if tuple(process['Edges']) == edge:
                            num_gpr += 1
                        else:
                            missing_process_edge.append(edge)

            if num_gpr == len(G.edges):
                print('All edges have Generate physical Bell pair process...') 
            else:
                print('Not all edge have Generate physical Bell pair process, the missing edges are, ')
                for edge in missing_process_edge:
                    print(edge)

            # Create process graph, process -> node , label -> edge

            modified_timeline = []
            for process in timeline:
                modified_process = {
                    'node': f'{process["Main Process"]}\n{tuple(process["Edges"])}',
                }
                modified_process = {**modified_process, **process}
                modified_timeline.append(modified_process)

            timelineGraph = nx.DiGraph()

            for node in modified_timeline:

                label_out = node['Label out']

                for next_node in modified_timeline:

                    if len(next_node['Edges']) == 2:
                    # Non-entanglement swapping process
                        if next_node['Label in'] == label_out and node['Edges'][0] == next_node['Edges'][0] and node['Edges'][-1] == next_node['Edges'][-1]:
                            timelineGraph.add_edge(node['node'], next_node['node'], label=label_out)
                    
                    else:
                    # Entanglement swapping process
                        if next_node['Label in'] == label_out and node['Edges'][0] in next_node['Edges'] and node['Edges'][-1] in next_node['Edges']:
                            timelineGraph.add_edge(node['node'], next_node['node'], label=label_out)

            # Check if limited process could reach all generate physical Bell pair with nodes.

            def check_reachable(process, modified_timeline, timelineGraph):
                reach_status = {}
                for node in process['Edges']:
                    reach_status[node] = False
                    for to_process in modified_timeline:
                        if to_process['Main Process'] in ['Generate physical Bell pair']:
                            try:
                                if nx.has_path(timelineGraph, to_process['node'], process['node']):
                                    reach_status[node] = True
                            except nx.NodeNotFound as e:
                                print(f"{e}")
                                return reach_status
                return reach_status
            
            limitedProcessReachStatus = {}
            for process in modified_timeline:
                if type(process['Num Trials']) is int:
                    
                    reach_status = check_reachable(process, modified_timeline, timelineGraph)
                    limitedProcessReachStatus[process['node']] = reach_status

            allGreen = False
            for limited_process in limitedProcessReachStatus:
                reachable = 0
                for node in limitedProcessReachStatus[limited_process]:
                    if limitedProcessReachStatus[limited_process][node]:
                        reachable += 1
                        # print(f'{limited_process} could reach {node}')
                    else:
                        pass
                        # print(f'{limited_process} could not reach {node}, this problem might coming from wrong node is specify to the process or/and label is not match.') 
                if reachable == len(limitedProcessReachStatus[limited_process]):
                    print(f'Limited process: {limited_process} is reachable to fundamental resource processes...')
                    allGreen = True
                else:
                    print(f'Limited process: {limited_process} is not reachable to fundamental resource processes, this might cause an error')
                    allGreen = False

                if allGreen:
                    pass
                    # if purification check number of external qubits

                    # if logical encoding check number of internal encoding qubits

            if allGreen:
                exper_allGreen += 1
                print(f'[{exper}] all status checked ')
            else:
                print(f'Checking is not pass, please be patient and re-check [{exper}] again.')


            net = Network( height='100%', width='100%', notebook=True) 


            # Code modified from https://gist.github.com/quadrismegistus/92a7fba479fc1e7d2661909d19d4ae7e
            # for each node and its attributes in the networkx graph
            for node, node_attrs in timelineGraph.nodes(data=True):
                net.add_node(str(node), shape='box', **node_attrs)
                
            # for each edge and its attributes in the networkx graph
            for source,target,edge_attrs in timelineGraph.edges(data=True):
                # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
                if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
                    # place at key 'value' the weight of the edge
                    edge_attrs['value']=edge_attrs['weight']
                # add the edge
                net.add_edge(str(source),str(target),**edge_attrs)

            # End modified code

            net.set_options(
                '''
                var options = {
                    "layout": {
                        "hierarchical": {
                        "enabled": true,
                        "levelSeparation": 250,
                        "direction": "LR",
                        "sortMethod": "directed"
                        }
                    },
                    "physics": {
                        "hierarchicalRepulsion": {
                        "centralGravity": 0
                        },
                        "minVelocity": 0.75,
                        "solver": "hierarchicalRepulsion"
                    }
                }
                '''
            )
            
            # net.show_buttons(filter_=['layout', 'interaction', 'physics'])
            self.process_graph[exper] = net
            if vis:
                net.show(f'{exper}-timeline.html')

            validate_row = {
                'Experiment': exper,
                'Resource-reachable': 'PASSED' if allGreen else 'FAILED',
            }
            for parameter in Parameters_test:
                validate_row[parameter] = 'PASSED' if Parameters_test[parameter] else 'FAILED'

            validate_table.append(validate_row)

        if get_table:
            df = pd.DataFrame(validate_table)
            dfStyler = df.style.set_properties(**{'text-align': 'left'})
            dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
            dfStyler.applymap(lambda v: "color:green;" if v == 'PASSED' else "")
            dfStyler.applymap(lambda v: "color:red;" if v == 'FAILED' else "")
            return dfStyler

        if exper_allGreen == len(self.strategies_list):
            print('All timeline and topology of all experiments are validated, you are good to execute Experiment.run() command!\
                 \nAnother error that is not currently check is the number of qubits needed to completed the task.')
            return True
        else:
            print('Test not passed.')
            return False

    def execute(self, multithreading=False, save_result=False):
        
        results = {}
        if multithreading:
            
            import ray
            ray.init()

            @ray.remote
            def single_execute(exp, save_result):
                return self.QuantumNetworks[exp].run(save_result=save_result)

            results = ray.get({exp : single_execute.remote(exp, save_result) for exp in self.strategies_list})
        else:
            for exp in self.strategies_list:
                result = self.QuantumNetworks[exp].run(save_result=save_result)
                results[exp] = result

        return results

class Tuner:

    def __init__(self, strategies):

        from bayes_opt import BayesianOptimization

        self.optimizers = {
            exp: BayesianOptimization(
                strategies[exp]['objective_function'],
                strategies[exp]['bounds'],
                verbose=0
            )
        for exp in strategies}

    def DistributedTuning(self, n_iter, path):

        import ray

        ray.init()

        @ray.remote
        def execute(optimizer, n_iter):
            self.optimizers[optimizer].maximize(n_iter=n_iter)
            data_tmp = []
            for res_dict in self.optimizers[optimizer].res:
                tmp = {param: res_dict['params'][param] for param in res_dict['params']}
                tmp['objective'] = res_dict['target']
                tmp['experiment'] = optimizer
                data_tmp.append(tmp)
            return data_tmp

        results = ray.get([execute.remote(optimizer, n_iter) for optimizer in self.optimizers])
        results = [item for sublist in results for item in sublist]
        results = pd.DataFrame(results)
        results.to_csv(path)

        return True

    def save_results(self, path):

        data = []
        for optimizer in self.optimizers:
            data_tmp = []
            for res_dict in self.optimizers[optimizer].res:
                tmp = {param: res_dict['params'][param] for param in res_dict['params']}
                tmp['objective'] = res_dict['target']
                tmp['experiment'] = optimizer
                data_tmp.append(tmp)
            data += data_tmp

        pd.DataFrame(data).to_csv(path)
        return pd.DataFrame(data)

class Experiment:
    def __init__(self, topologies, timelines, nodes_info=None, memFunc=None, gateError=None, simTime=None, measurementError=None,
                 parameters_set=None, label_records=None, BSA_prob=None,
                 path='experiment', message_log='exp', repeat=5,
                 progress_bar=False, collect_fidelity_history=False) -> None:

        self.experiments = list(timelines.keys())
        self.topologies = {}
        self.timelines = {}
        self.memFunc = {}
        self.gateError = {}
        self.measurementError = {}
        self.simTime = {}
        self.nodes_info = {}
        self.label_recorded = {}
        self.process_graph = {}
        self.progress_bar = not progress_bar

        if label_records is None:
            label_records = {exper: 'Physical' for exper in self.experiments}

        for exper in self.experiments:
            self.topologies[exper] = [topologies[exper] for _ in range(len(parameters_set))]
            self.timelines[exper] = [timelines[exper] for _ in range(len(parameters_set))]
            self.memFunc[exper] = [memFunc[exper] for _ in range(len(parameters_set))]
            self.gateError[exper] = [gateError[exper] for _ in range(len(parameters_set))]
            self.measurementError[exper] = [measurementError[exper] for _ in range(len(parameters_set))]
            self.simTime[exper] = [simTime[exper] for _ in range(len(parameters_set))]
            self.label_recorded[exper] = label_records[exper]

            if nodes_info is not None:
                self.nodes_info[exper] = [nodes_info[exper] for _ in range(len(parameters_set))]

        self.parameters_set = parameters_set
        self.BSA_prob = BSA_prob
        self.path = path
        self.message_log = message_log
        self.repeat = repeat
        self.configurations = {}
        self.executed = False
        self.Fidelity = {}
        self.Time_used = {} # Simulation time used to complete unlimited process
        
        for exper in self.experiments:
            self.configurations[exper] = []
            for index in range(len(parameters_set)):
                tmp_list = []
                for rd in range(repeat):
                    config = Configuration(self.topologies[exper][index], self.timelines[exper][index],
                                           nodes_info=self.nodes_info[exper][index],
                                           memFunc=self.memFunc[exper][index],
                                           gate_error=self.gateError[exper][index],
                                           measurementError=self.measurementError[exper][index],
                                           sim_time=self.simTime[exper][index],
                                           result_path=self.path,
                                           message=f'exp{exper}_p{self.parameters_set[index]}_r{rd}',
                                           collectFidelityHistory=collect_fidelity_history,
                                           experiment=self.message_log, label_record=self.label_recorded[exper])
                    tmp_list.append(config)
                self.configurations[exper].append(tmp_list)
                
    def run(self, save_tomo=False, save_result=True):

        for exper in tqdm(self.experiments, desc='Experiments executed: ',disable=self.progress_bar):
            self.Fidelity[exper] = []
            self.Time_used[exper] = []
            for index in tqdm(range(len(self.parameters_set)), desc='Parameters executed: ',disable=self.progress_bar, leave=False):
                tmp_list_fidelity = []
                tmp_list_time_used = []
                for rd in range(self.repeat):
                    Q = QuantumNetwork(configuration=self.configurations[exper][index][rd])
                    result = Q.run(save_tomography=save_tomo, save_result=save_result)
                    tmp_list_fidelity.append(result['fidelity'])
                    tmp_list_time_used.append(result['Time used'])
                self.Fidelity[exper].append(tmp_list_fidelity)
                self.Time_used[exper].append(tmp_list_time_used)

        self.executed = True

    def validate_run(self):

        # TODO implement validate checking system before actual execution
        passed = True

        return passed

    def validate(self, vis=False, get_table=False):

        validate_table = []

        exper_allGreen = 0
        for exper in self.experiments:

            topology = self.topologies[exper][0]
            timeline = self.timelines[exper][0]
            sim_time = self.simTime[exper][0]
            gate_error = self.gateError[exper][0]
            measurement_error = self.measurementError[exper][0]
            memory_error = self.memFunc[exper][0]
            nodes_info = {}
            for node in self.nodes_info[exper][0]:
                if node not in ['numPhysicalBuffer', 'numInternalEncodingBuffer', 'numInternalDetectingBuffer', 'numInternalInterfaceBuffer']:
                    nodes_info[node] = self.nodes_info[exper][0][node]

            # Detect dynamic nodes
            Dynamic_flag = False
            for node in nodes_info:
                if callable(nodes_info[node]['coordinate']):
                    Dynamic_flag = True

            # Validate parameter
            Parameters_test = {
                'loss': True,
                'depolarizing error': True,
                'gate error': True,
                'memory error': True,
                'measurement error': True,
            }

            # loss and depolarizing_error
            for edge in topology:
                loss = topology[edge]['loss'] # dB/km
                if type(loss) is not float and type(loss) is not int:
                    Parameters_test['loss'] = False

                p_dep = topology[edge]['depolarlizing error']
                if type(p_dep) is list:
                    if len(p_dep) != 4:
                        print(f'[{exper}] length depolarizing probability of {edge} is not 4')
                        Parameters_test['depolarizing error'] = False
                    if int(round(np.abs(sum(p_dep)), 4)) != 1:
                        print(f'[{exper}] WARNING sum of depolarizing probability of {edge} is not 1 to decimal 4.')
                        Parameters_test['depolarizing error'] = False

            # gate_error
            if len(gate_error) != 2:
                print(f'[{exper}] length of gate error probability of {edge} is not 2')
                Parameters_test['gate error'] = False
            if int(round(sum(gate_error), 4)) != 1:
                print(f'[{exper}] WARNING sum of gate error probability of {edge} is not 1 to decimal 4.')

            # memory_error
            if callable(memory_error):
                for i in range(10):
                    x = random.random()
                    p = memory_error(x)
                    if len(p) != 4:
                        print(f'[{exper}] length of memory error is 4')
                        Parameters_test['memory error'] = False
                    if int(round(sum(p), 2)) != 1:
                        print(f'[{exper}] WARNING sum of memory error probability of {edge} is not 1 to decimal 2.')
            else:
                print(f'[{exper}] WARNING memory error is not callable function')
                if len(memory_error) != 4:
                    print(f'[{exper}] length of memory error is 4')
                    Parameters_test['memory error'] = False
            
            # measurement_error
            if measurement_error > 1:
                print(f'[{exper}] measurement error provided is more than 1')
                Parameters_test['measurement error'] = False

            # Valiate Process-flow and number of resource

            G = nx.Graph()
            for edge in topology:
                G.add_edge(edge[0], edge[1]) 
            nx.set_edge_attributes(G, topology)

            print('Limited Process: ')
            limited_processes = []
            for process in timeline:
                if type(process['Num Trials'] )is int:
                    limited_processes.append(process['Main Process'])

                    if tuple(process['Edges']) == (list(G.nodes())[0], list(G.nodes())[-1]):
                        is_end_to_end = f'End-to-end process -> {process["Edges"]}'
                    else:
                        is_end_to_end = f'Non-E2E process -> {process["Edges"]}'

                    print(f'{len(limited_processes)}. ' ,process['Main Process'], f': {is_end_to_end}')
            

            if len(limited_processes) == 0:
            # Detect no limited process simulation
                print(f'[{exper}] experiment has no limited process...')

                if sim_time is None:
                    print(f'[{exper}] WARNING: simulation might not terminate')
            
            if sim_time is not None:
                print(f'[{exper}] experiment will end with simulation time limit: {sim_time}s')

            if Dynamic_flag:
                print(f'[{exper}] experiment contains dynamic nodes...')
                if sim_time is None:
                    print(f'[{exper}] experiment will terminate only if limited process is finish, please make sure that loss and distance provided are reasonable as simulation might take insanely much time to finish...')

            if nx.is_connected(G):
                print('Topology provied is connected graph...')
            else:
                print('The topology provided is not connected, this might not be what is expected...')

            # Validate if all edges have generate physical resource

            num_gpr = 0
            missing_process_edge = []
            for edge in G.edges:
                
                for process in timeline:
                    if process['Main Process'] in ['Generate physical Bell pair']:
                        if tuple(process['Edges']) == edge:
                            num_gpr += 1
                        else:
                            missing_process_edge.append(edge)

            if num_gpr == len(G.edges):
                print('All edges have Generate physical Bell pair process...') 
            else:
                print('Not all edge have Generate physical Bell pair process, the missing edges are, ')
                for edge in missing_process_edge:
                    print(edge)

            # Create process graph, process -> node , label -> edge

            modified_timeline = []
            for process in timeline:
                modified_process = {
                    'node': f'{process["Main Process"]}\n{tuple(process["Edges"])}',
                }
                modified_process = {**modified_process, **process}
                modified_timeline.append(modified_process)

            timelineGraph = nx.DiGraph()

            for node in modified_timeline:

                label_out = node['Label out']

                for next_node in modified_timeline:

                    if len(next_node['Edges']) == 2:
                    # Non-entanglement swapping process
                        if next_node['Label in'] == label_out and node['Edges'][0] == next_node['Edges'][0] and node['Edges'][-1] == next_node['Edges'][-1]:
                            timelineGraph.add_edge(node['node'], next_node['node'], label=label_out)
                    
                    else:
                    # Entanglement swapping process
                        if next_node['Label in'] == label_out and node['Edges'][0] in next_node['Edges'] and node['Edges'][-1] in next_node['Edges']:
                            timelineGraph.add_edge(node['node'], next_node['node'], label=label_out)

            # Check if limited process could reach all generate physical Bell pair with nodes.

            def check_reachable(process, modified_timeline, timelineGraph):
                reach_status = {}
                for node in process['Edges']:
                    reach_status[node] = False
                    for to_process in modified_timeline:
                        if to_process['Main Process'] in ['Generate physical Bell pair']:
                            try:
                                if nx.has_path(timelineGraph, to_process['node'], process['node']):
                                    reach_status[node] = True
                            except nx.NodeNotFound as e:
                                print(f"{e}")
                                return reach_status
                return reach_status
            
            limitedProcessReachStatus = {}
            for process in modified_timeline:
                if type(process['Num Trials']) is int:
                    
                    reach_status = check_reachable(process, modified_timeline, timelineGraph)
                    limitedProcessReachStatus[process['node']] = reach_status

            allGreen = False
            for limited_process in limitedProcessReachStatus:
                reachable = 0
                for node in limitedProcessReachStatus[limited_process]:
                    if limitedProcessReachStatus[limited_process][node]:
                        reachable += 1
                        # print(f'{limited_process} could reach {node}')
                    else:
                        pass
                        # print(f'{limited_process} could not reach {node}, this problem might coming from wrong node is specify to the process or/and label is not match.') 
                if reachable == len(limitedProcessReachStatus[limited_process]):
                    print(f'Limited process: {limited_process} is reachable to fundamental resource processes...')
                    allGreen = True
                else:
                    print(f'Limited process: {limited_process} is not reachable to fundamental resource processes, this might cause an error')
                    allGreen = False

                if allGreen:
                    pass
                    # if purification check number of external qubits

                    # if logical encoding check number of internal encoding qubits

            if allGreen:
                exper_allGreen += 1
                print(f'[{exper}] all status checked ')
            else:
                print(f'Checking is not pass, please be patient and re-check [{exper}] again.')


            net = Network( height='100%', width='100%', notebook=True)

            # Code modified from https://gist.github.com/quadrismegistus/92a7fba479fc1e7d2661909d19d4ae7e
            # for each node and its attributes in the networkx graph
            for node, node_attrs in timelineGraph.nodes(data=True):
                net.add_node(str(node), shape='box', **node_attrs)
                
            # for each edge and its attributes in the networkx graph
            for source,target,edge_attrs in timelineGraph.edges(data=True):
                # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
                if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
                    # place at key 'value' the weight of the edge
                    edge_attrs['value']=edge_attrs['weight']
                # add the edge
                net.add_edge(str(source),str(target),**edge_attrs)

            # End modified code
            net.set_options(
                '''
                var options = {
                    "layout": {
                        "hierarchical": {
                        "enabled": true,
                        "levelSeparation": 250,
                        "direction": "LR",
                        "sortMethod": "directed"
                        }
                    },
                    "physics": {
                        "hierarchicalRepulsion": {
                        "centralGravity": 0
                        },
                        "minVelocity": 0.75,
                        "solver": "hierarchicalRepulsion"
                    }
                }
                '''
            )
            
            # net.show_buttons(filter_=['layout', 'interaction', 'physics'])
            self.process_graph[exper] = net
            if vis:
                net.show(f'{exper}-timeline.html')

            validate_row = {
                'Experiment': exper,
                'Resource-reachable': 'PASSED' if allGreen else 'FAILED',
            }
            for parameter in Parameters_test:
                validate_row[parameter] = 'PASSED' if Parameters_test[parameter] else 'FAILED'

            validate_table.append(validate_row)

        if get_table:
            df = pd.DataFrame(validate_table)
            dfStyler = df.style.set_properties(**{'text-align': 'left'})
            dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
            return dfStyler

        if exper_allGreen == len(self.experiments):
            print('All timeline and topology of all experiments are validated, you are good to execute Experiment.run() command! \nAnother error that is not currently check is the number of qubits needed to completed the task.')
            return True
        else:
            print('Test not passed.')
            return False

    def get_fidelity(self, read_from=None):

        if self.executed is False and read_from is None:
            print('The experiment is not yet execute, please execute experiment.run() or provide result files')
            return None

        if read_from is not None:
            Fidelity = {}
            for exper in self.experiments:
                Fidelity[exper] = []
                for index in range(len(self.parameters_set)):
                    tmp_list = []
                    for rd in range(self.repeat):
                        with open(f"{read_from}/Result_{self.message_log}_exp{exper}_p{self.parameters_set[index]}_r{rd}.pkl", "rb") as f:
                            exp = dill.load(f)
                        tmp_list.append(exp['fidelity'])
                    Fidelity[exper].append(tmp_list)
            
            return Fidelity
        
        return self.Fidelity


    def getDataFrame(self, read_from=None):

        if read_from is not None:
            Fidelity = self.get_fidelity(read_from=read_from)
        else:
            Fidelity = self.get_fidelity()

        df = []
        for index in range(len(self.parameters_set)):
            for rd in range(self.repeat):
                tmp = {exper: Fidelity[exper][index][rd] for exper in self.experiments}
                tmp['Probability (%)'] = self.parameters_set[index]
                df.append(tmp)
        
        return pd.DataFrame(df)

    def results(self):

        results = {}
        for exper in self.experiments:
            results[exper] = {}
            for index in range(len(self.parameters_set)):
                results[exper][index] = {}
                for rd in range(self.repeat):
                    with open(f"{self.path}/Result_{self.message_log}_exp{exper}_p{self.parameters_set[index]}_r{rd}.pkl", "rb") as f:
                        exp = dill.load(f)
                    results[exper][index][rd] = exp

        return results

    def raw_tomo_results(self):

        results = {}
        for exper in self.experiments:
            results[exper] = {}
            for index in range(len(self.parameters_set)):
                results[exper][index] = {}
                for rd in range(self.repeat):
                    results[exper][index][rd] = pd.read_csv(f"{self.path}/StateTomography_{self.message_log}_exp{exper}_p{self.parameters_set[index]}_r{rd}.csv")[['qubit1', 'qubit2']].to_dict('records')

        return results

    def plot_Expected_waiting_time(self, ):

        if self.executed is False:
            results = self.results()
        else:
            results = {}
            for exper in self.Time_used:
                results[exper] = {}
                for index in range(len(self.BSA_prob)):
                    results[exper][index] = {}
                    for rd in range(self.repeat):
                        results[exper][index][rd] = {'Time used': self.Time_used[exper][index][rd]}

        wt = {}
        for exper in results:
            wt[exper] = {}
            Prob_List = []
            WT_list = []
            for index in range(len(self.BSA_prob)):
                wt[exper][index] = {}
                for rd in range(self.repeat):
                    wt[exper][index][rd] = results[exper][index][rd]['Time used']

                    Prob_List.append(self.BSA_prob[index])
                    WT_list.append(wt[exper][index][rd])
    
            df = pd.DataFrame({'prob': Prob_List, 'trials': WT_list})
            sns.lineplot(data=df, x='prob', y='trials', ci='sd' ,label=exper)

        return 

    def plot_fidelity_history(self, figsize=None, row_col=None, analytic_fidelity=None):

        if analytic_fidelity is None:
            analytic_fidelity = {
                exper: {
                    index: 0
                for index in range(len(self.parameters_set))}
            for exper in self.experiments}

        # Extract information from result
        results = self.results()
        fidelity_hist = {}
        for exper in results:
            fidelity_hist[exper] = {}
            for index in range(len(self.parameters_set)):
                fidelity_hist[exper][index] = []
                for meas in range(len(results[exper][index][0]['Fidelity History'])):
                    for rd in range(self.repeat):
                        value = abs(results[exper][index][rd]['Fidelity History'][meas] - analytic_fidelity[exper][index])*100
                        tmp = {'meas': meas, f'rd': float(value)}
                        fidelity_hist[exper][index].append(tmp)
        # return fidelity_hist
        # Plotting
        sns.set_theme()
        if row_col is None:
            row_col = (1, len(self.experiments))
        if figsize is None:
            figsize = (int(18*row_col[0]), int(5*row_col[1]))
        fig, axes = plt.subplots(row_col[0], row_col[1], figsize=figsize)
        for row in range(row_col[0]):
            for col in range(row_col[1]):
                for index in range(len(self.parameters_set)):
                    exper = self.experiments[int(row_col[0]*row + col)]
                    text = f'paramters: {index}'
                    if row_col[0] == 1:
                        if row_col[1] == 1:
                            sns.lineplot(ax=axes, x='meas', y='rd', data=pd.DataFrame(fidelity_hist[exper][index]), ci='sd', label=text)
                            axes.set_title(exper)
                        else:
                            sns.lineplot(ax=axes[col], x='meas', y='rd', data=pd.DataFrame(fidelity_hist[exper][index]), ci='sd', label=text)
                            axes[col].set_title(exper)
                    else:
                        sns.lineplot(ax=axes[row, col], x='meas', y='rd', data=pd.DataFrame(fidelity_hist[exper][index]), ci='sd', label=text)
                        axes[row, col].set_title(exper)

        return fig, axes

    def plot(self, read_from=None):

        df = self.getDataFrame(read_from=read_from)

        sns.set_theme()
        for exper in self.experiments:
            fig = sns.lineplot(x='Probability (%)',y=exper, data=df, ci='sd', label=exper, markers=True, dashes=False)
        plt.ylabel('Fidelity')
        return fig

    @staticmethod
    def read_timeline_from_csv(file_path, excel=False, sheet_name=0):
        if excel == True:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_csv(file_path)
        df['Edges'] = df['Edges'].transform(lambda x: ast.literal_eval(x))
        df['Num Trials'] = df['Num Trials'].transform(lambda x: True if x == 'True' else x)
        df['Num Trials'] = df['Num Trials'].transform(lambda x: int(x) if int(x) > 1 else x)
        return df.to_dict('records') ,df

    def str_to_list(x):
        return ast.literal_eval(x)


class Configuration:
    def __init__(self, topology, timeline, experiment=False, nodes_info=None, memFunc=None, gate_error=None, measurementError=None,
                 message=None, throughtput_edges=None, label_record='Physical', num_measured=9000, sim_time=None, 
                 collectFidelityHistory=False, result_path='result'):

        self.numPhysicalBuffer = 20
        self.numInternalEncodingBuffer = 20
        self.numInternalDetectingBuffer = 10
        self.numInternalInterfaceBuffer = 2
        self.memFunc = np.inf if memFunc is None else memFunc # Function of memory of qubit
        self.gateError = 0 if gate_error is None else gate_error
        self.measurementError = 0 if measurementError is None else measurementError
        self.timeline = timeline
        self.experiment = experiment
        self.light_speed_in_fiber = 208189.206944 # km/s
        self.message = message
        self.num_measured = num_measured
        self.g = topology
        self.result_path = result_path
        self.label_recorded = label_record
        self.collectFidelityHistory = collectFidelityHistory
        self.simulation_time = sim_time
        self.coor_system = 'normal'
        
        # Initialize graph
        G = nx.Graph()
        for edge in topology:

            for node in edge:

                if node not in topology.keys():
                    topology[edge][node] = {
                        'memory function': self.memFunc,
                        'gate error': self.gateError,
                        'measurement error': self.measurementError
                    }

                if not callable(topology[edge][node]['memory function']):
                    memory_time = topology[edge][node]['memory function']
                    
                    # Memory error
                    def memory_error_function(time, tau=memory_time):
                        p = 0.75*np.e**(-1*(time/tau)) + 0.25
                        return [p, (1- p)/3, (1- p)/3, (1- p)/3]

                    topology[edge][node]['memory function'] = memory_error_function

            G.add_edge(edge[0], edge[1]) 

        nx.set_edge_attributes(G, topology)
        

        # Include function of error model
        if nodes_info is not None:
            nx.set_node_attributes(G, nodes_info)
            self.nodes_info = nodes_info
            self.numPhysicalBuffer = self.nodes_info['numPhysicalBuffer']
            self.numInternalEncodingBuffer = self.nodes_info['numInternalEncodingBuffer']
            self.numInternalDetectingBuffer = self.nodes_info['numInternalDetectingBuffer']
            self.numInternalInterfaceBuffer = self.nodes_info['numInternalInterfaceBuffer']

        self.NetworkTopology = G
        if throughtput_edges is None:
            self.throughtputEdges = [list(G.nodes())[0], list(G.nodes())[-1]]
        else:
            self.throughtputEdges = throughtput_edges


class QuantumNetwork(_GeneratePhyscialResource.Mixin, 
                     _EntanglementPurification.Mixin, 
                     _EntanglementSwapping.Mixin, 
                     _GenerateLogicalResource.Mixin, 
                     _VirtualStateTomography.Mixin,
                     _TimeLag.Mixin):

    def __init__(self, configuration):

        self.configuration = configuration
        self.env = simpy.Environment()

        # Initialize QNICs in each nodes
        self.graph = self.configuration.NetworkTopology
        self.complete_graph = nx.complete_graph(self.graph.nodes)
        self.edges_list = [f'{node1}-{node2}' for node1, node2 in list(self.graph.edges())]
        self.complete_edges_list = [f'{node1}-{node2}' for node1, node2 in list(self.complete_graph.edges())]
        self.QuantumChannel = {edge: simpy.Store(self.env) for edge in self.edges_list}
        self.ClassicalChannel = {edge: simpy.Store(self.env) for edge in self.edges_list}

        self.table_name = {
            'externalQubitsTable': self.configuration.numPhysicalBuffer,
            'externalBusyQubitsTable': 0,
            'internalEncodingQubitTable': self.configuration.numInternalEncodingBuffer,
            'internalDetectingQubitTable': self.configuration.numInternalDetectingBuffer,
            'internalInterfaceQubitTable': self.configuration.numInternalInterfaceBuffer
        }

        self.QubitsTables = {
             table : { f'{node1}-{node2}': {
                f'QNICs-{node1}' : simpy.FilterStore(self.env),
                f'QNICs-{node2}' : simpy.FilterStore(self.env)
            } for node1, node2 in self.graph.edges() }
        for table in self.table_name}

        for table in self.table_name:
            for node1, node2, attr in self.graph.edges(data=True): # self.complete_graph.edges() #  
                for i in range(self.table_name[table]):

                    self.QubitsTables[table][f'{node1}-{node2}'] \
                    [f'QNICs-{node1}'].put(PhysicalQubit(node1, i, f'{node1}-{node2}', table[:8], self.env, table, attr[node1]['memory function'], attr[node1]['gate error'], attr[node1]['measurement error']))
                    self.QubitsTables[table][f'{node1}-{node2}'] \
                    [f'QNICs-{node2}'].put(PhysicalQubit(node2, i, f'{node1}-{node2}', table[:8], self.env, table, attr[node2]['memory function'], attr[node2]['gate error'], attr[node2]['measurement error']))

        self.internalLogicalQubitTable = { f'{node1}-{node2}': {
            f'QNICs-{node1}' : [],
            f'QNICs-{node2}' : []
        } for node1, node2 in self.graph.edges() }

        self.resource_table_name = [
            'physicalResourceTable', 
            'internalPhysicalResourceTable', 
            'internalPurifiedResourceTable',
            'internalSecondPurifiedResourceTable',
            'logicalResourceTable'
        ]

        self.resourceTables = {}
        for table in self.resource_table_name:
            self.resourceTables[table] = {}
            for node1, node2 in self.complete_graph.edges():
                if table[:8] == 'internal':
                    self.resourceTables[table][f'{node1}-{node2}'] = {
                        f'{node1}' : simpy.FilterStore(self.env),
                        f'{node2}' : simpy.FilterStore(self.env),
                    }
                else:
                    self.resourceTables[table][f'{node1}-{node2}'] = simpy.FilterStore(self.env) 

        #self.complete_edges_list = list(set(self.complete_edges_list))
        #self.edges_list = list(set(self.edges_list))
        for process in self.configuration.timeline:
            process['isSuccess'] = 0

        self.simulationLog = [] # <= TODO use this to collect data to plot
        self.qubitsLog = []
        self.numResrouceProduced = 0
        self.numBaseBellAttempt = 0
        self.numResourceUsedForFidelityEstimation = 0
        
        # For fidelity calculation
        self.measurementResult = []
        self.fidelityHistory = []
        self.Expectation_value = {
            'XX': {'commute': 0, 'anti-commute': 0},
            'YY': {'commute': 0, 'anti-commute': 0},
            'ZZ': {'commute': 0, 'anti-commute': 0}
        }
        
        self.fidelityStabilizerMeasurement = None

    '''
    Ulits
    '''

    def updateLog(self, log_message):
        self.simulationLog.append(log_message)
        return None

    def createLinkResource(self, node1, node2, resource1, resource2, resource_table, label='Physical', target_node=None, initial=False):
        resource1.isBusy, resource2.isBusy = True, True
        resource1.partner, resource2.partner = resource2, resource1
        resource1.partnerID, resource2.partnerID = resource2.qubitID, resource1.qubitID
        '''
        if initial == True:
            if resource1.initiateTime is not None or resource2.initiateTime is not None:
                raise ValueError('This qubit has not set free properly')
            resource1.initiateTime, resource2.initiateTime = self.env.now, self.env.now
        '''
        if label == 'Internal':
            resource1.setInitialTime()
            resource2.setInitialTime()
            resource_table[f'{node1}-{node2}'][f'{target_node}'].put((resource1, resource2, 'Internal'))
            return None
        
        resource_table[f'{node1}-{node2}'].put((resource1, resource2, label))
        # self.updateLog({'Time': self.env.now, 'Message': f'Qubit ({resource1.qubitID}) entangle with Qubit ({resource2.qubitID})'})

        if (node1 in self.configuration.throughtputEdges) and (node2 in self.configuration.throughtputEdges) and (label == self.configuration.label_recorded):
            self.numResrouceProduced += 1

        # self.updateLog({'Time': self.env.now, 'Message': f'Qubit ({resource1.qubitID}) entangle with Qubit ({resource2.qubitID})'})

        return None

    def validateNodeOrder(self, node1, node2):

        if f'{node1}-{node2}' not in self.complete_edges_list:
            node1, node2 = node2, node1

        return node1, node2

    def Timeline(self):

        connectionSetup = [self.env.process(self.ConnectionSetup(self.configuration.throughtputEdges[0], 
                                                                 self.configuration.throughtputEdges[1]))]
        yield simpy.AllOf(self.env, connectionSetup)

        Unlimited_process = []
        Limited_process = []
        for process in self.configuration.timeline:
            if process['Main Process'] == '_GeneratePhysicalResource':
                p = [self.env.process(self.generatePhysicalResource(process['Edges'][0], process['Edges'][1], 
                                                                   label_out=process['Label out'],
                                                                   num_required=process['Num Trials']))]
            elif process['Main Process'] == '_EntanglementSwapping':
                p = [self.env.process(self.ExternalEntanglementSwapping( (process['Edges'][0], process['Edges'][1]), (process['Edges'][1], process['Edges'][2]), \
                                                                         num_required=process['Num Trials'], \
                                                                         label_in=process['Label in'], \
                                                                         label_out=process['Label out'], \
                                                                         resource_type=process['Resource Type']))]
            elif process['Main Process'] == '_StateTomography':
                p = [self.env.process(self.VirtualStateTomography(process['Edges'][0], process['Edges'][1], \
                                                                 num_required=process['Num Trials'], \
                                                                 label_in=process['Label in'], \
                                                                 resource_type=process['Resource Type']))]
            elif process['Main Process'] == '_Purification':
                p = [self.env.process(self.Purification(process['Edges'][0], process['Edges'][1], 
                                                       num_required=process['Num Trials'], \
                                                       label_in=process['Label in'], \
                                                       label_out=process['Label out'], \
                                                       protocol=process['Protocol']))]
            elif process['Main Process'] == '_GenerateLogicalResource':
                p = [self.env.process(self.generateLogicalResource(process['Edges'][0], process['Edges'][1], 
                                                                  num_required=process['Num Trials'], \
                                                                  label_in=process['Label in'], \
                                                                  label_out=process['Label out'], \
                                                                  protocol=process['Protocol']))]
            elif process['Main Process'] == '_GeneratePhysicalResourceEPPS':
                p = [self.env.process(self.generatePhysicalResourceEPPS(process['Edges'][0], process['Edges'][2], process['Edges'][1],
                                                                               label_out=process['Label out'],
                                                                               num_required=process['Num Trials']))]
            elif process['Main Process'] == '_GeneratePhysicalResourceEPPSPulse':
                p = [self.env.process(self.generatePhysicalResourceEPPSPulse(process['Edges'][0], process['Edges'][2], process['Edges'][1],
                                                                                    label_out=process['Label out'],
                                                                                    num_required=process['Num Trials']))]
            elif process['Main Process'] == '_GeneratePhysicalResourcePulse':
                p = [self.env.process(self.generatePhysicalResourceStaticPulse(process['Edges'][0], process['Edges'][1], 
                                                                            label_out=process['Label out'],
                                                                            num_required=process['Num Trials']))]
            elif process['Main Process'] in ['PrototypeGeneratePhysicalResourcePulse', 'Generate physical Bell pair']: 
                p = [
                    self.env.process(self.Emitter(process['Edges'][0], process['Edges'][1], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'])),
                    self.env.process(self.Detector(process['Edges'][0], process['Edges'][1], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'])),
                    self.env.process(self.ClassicalMessageHandler(process['Edges'][0], process['Edges'][1], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials']))
                ]
            elif process['Main Process'] in ['PrototypeGeneratePhysicalResourcePulseMidpoint', 'Generate physical Bell pair (Midpoint)']: 
                p = [
                    self.env.process(self.Emitter(process['Edges'][0], process['Edges'][2], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'], middleNode=process['Edges'][1], EPPS=process['Protocol'])),
                    self.env.process(self.Detector(process['Edges'][0], process['Edges'][2], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'], middleNode=process['Edges'][1], EPPS=process['Protocol'])),
                    self.env.process(self.ClassicalMessageHandler(process['Edges'][0], process['Edges'][2], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'], middleNode=process['Edges'][1]))
                ]
            elif process['Main Process'] in ['PrototypeEntanglementSwapping', 'Entanglement swapping']:
                p = [self.env.process(self.PrototypeExternalEntanglementSwapping(process,( process['Edges'][0], process['Edges'][1]), (process['Edges'][1], process['Edges'][2]), \
                                                                                num_required=process['Num Trials'], \
                                                                                label_in=process['Label in'], \
                                                                                label_out=process['Label out'], \
                                                                                resource_type=process['Resource Type'],
                                                                                note=process['Note']))]    
            elif process['Main Process'] in ['PrototypeStateTomography', 'State tomography']:
                p = [self.env.process(self.PrototypeVirtualStateTomography(process, process['Edges'][0], process['Edges'][1], \
                                                                        num_required=process['Num Trials'], \
                                                                        label_in=process['Label in'], \
                                                                        resource_type=process['Resource Type'],
                                                                        note=process['Note']))] 
            elif process['Main Process'] in ['PrototypePurification', 'Entanglement purification']:
                p = [self.env.process(self.PrototypePurification(process, process['Edges'][0], process['Edges'][1], 
                                                       num_required=process['Num Trials'], \
                                                       label_in=process['Label in'], \
                                                       label_out=process['Label out'], \
                                                       protocol=process['Protocol'],
                                                       note=process['Note']))]   
            elif process['Main Process'] in ['PrototypeGenerateLogicalResource', 'Generate logical Bell pair']:
                p = [self.env.process(self.PrototypeGenerateLogicalResource(process, process['Edges'][0], process['Edges'][1], 
                                                                  num_required=process['Num Trials'], \
                                                                  label_in=process['Label in'], \
                                                                  label_out=process['Label out'], \
                                                                  protocol=process['Protocol']))]                                                                                              
            else:
                raise ValueError('Process is not define.')
            if type(process['Num Trials']) is int:
                Limited_process += p
            else:
                Unlimited_process += p

        yield simpy.AllOf(self.env, Limited_process)

        #connectionSetup = [self.env.process(self.ConnectionSetup(self.configuration.throughtputEdges[0], 
        #                                                         self.configuration.throughtputEdges[1]))]
        #yield simpy.AllOf(self.env, connectionSetup)

        self.connectionSetupTimeStamp = self.env.now - self.connectionSetupTimeStamp

    def run(self, save_tomography=False, save_result=True):

        timeline = self.env.process(self.Timeline())
        sim_time = timeline if self.configuration.simulation_time is None else self.configuration.simulation_time
        self.env.run(until=sim_time)

        # Store simulated data 

        # Create folder if not exist
        if save_tomography is True or save_result is True:
            if (not os.path.exists(f'{self.configuration.result_path}')):
                os.makedirs(f'{self.configuration.result_path}')
        
        if save_tomography is True:
            # Data for state-tomography
            pd.DataFrame(self.measurementResult).to_csv(f'{self.configuration.result_path}/StateTomography_{self.configuration.experiment}_{self.configuration.message}.csv')

        config = {data: self.configuration.__dict__[data] for data in self.configuration.__dict__}
        config['fidelity'] = self.fidelityStabilizerMeasurement
        config['Resources Produced'] = self.numResrouceProduced
        config['Base Resources Attempt'] = self.numBaseBellAttempt
        config['Resource Used in Fidelity Estimation'] = self.numResourceUsedForFidelityEstimation
        config['Time used'] = self.connectionSetupTimeStamp
        config['Fidelity History'] = self.fidelityHistory
        config['Qubits waiting time'] = self.qubitsLog

        # Save log data
        config['Simulation log'] = self.simulationLog

        if save_result is True:
            with open(f"{self.configuration.result_path}/Result_{self.configuration.experiment}_{self.configuration.message}.pkl", "wb") as f:
                dill.dump(config, f)

        return config