from optparse import Option
import os 
from typing import List, Dict, Union, Any, Callable, Optional
import random
import ast

import simpy
import networkx as nx
import numpy as np
import pandas as pd
import dill
from pyvis.network import Network

from .Qubit.qubit import PhysicalQubit, LogicalQubit
from .QuantumProcess import _EntanglementPurification, _EntanglementSwapping, _GenerateLogicalResource, _GeneratePhyscialResource, _FidelityEstimation
from .SubProcess import _TimeLag


class Xperiment:

    def __init__(self, timelines_path: str, 
                       nodes_info_exp: Dict, 
                       edges_info_exp: Dict, 
                       gate_error: Union[int, float] = 0,
                       measurement_error: Union[int, float] = 0,
                       memory_time: Union[int, float] = 1,
                       strategies_list: Optional[List] = None, 
                       resources_dict: Optional[Dict] = None,
                       experiment: Optional[Any] = None,
                       sim_time: Union[int, float] = None,
                ):
        """Init of Xperiment

        Args:
            timelines_path (str): Path to excel file containing resource-process flow
            nodes_info_exp (Dict): Information of nodes in network topology
            edges_info_exp (Dict): Information of edges in network topology
            gate_error (Union[int, float], optional): Probablity of gate error to be validate. Defaults to 0.
            measurement_error (Union[int, float], optional): Probablity of measurement error to be validate. Defaults to 0.
            memory_time (Union[int, float], optional): Memory time of qubit to be validate. Defaults to 1.
            strategies_list (Optional[List], optional): List of strategies to be simulated as a sheet names. Defaults to None.
            resources_dict (Optional[Dict], optional): Dictionary of resource in each QNICs. Defaults to None.
            experiment (Optional[Any], optional): Name of file for result to be added. Defaults to None.
            sim_time (Union[int, float], optional): Time in simulation allowed for simulation to execute for each strategies. Defaults to None.
        """

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

        self.label_records = {}
        for exp in self.strategies_list:
            label_record = []
            for process in self.timelines[exp]:
                label_record.append(process['Label out'])
            self.label_records[exp] = list(set(label_record))
        
        self.configurations = {
            exp : Configuration(
                edges_info = self.edges_info_exp[exp], 
                timeline = self.timelines[exp], 
                nodes_info = self.nodes_info_exp[exp], 
                memory_function = self.memory_functions[exp], 
                gate_error = self.gate_errors[exp], 
                measurement_error = self.measurement_errors[exp],
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

    def validate(self, 
                 vis: Optional[bool] = False, 
                 get_table: Optional[bool] = False, 
                 show_message: Optional[bool] = False):
        """Validation method, checking for error models, resource-process flow. 
           Currently not validate minimum resources needed for execute.

        Args:
            vis (Optional[bool], optional): Save html resource-process graph. Defaults to False.
            get_table (Optional[bool], optional): Show summary table. Defaults to False.
            show_message (Optional[bool], optional): Print validation message. Defaults to False.

        Returns:
            bool: If validation is passed or not.
        """

        validate_table = []
        message = ''

        exper_allGreen = 0
        for exper in self.strategies_list:

            message += f'\nValidating experiment: {exper}'
            message += '\n---------------------------------------------'

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
                        message += f'\n[{exper}] length depolarizing probability of {edge} is not 4'
                        Parameters_test['depolarizing error'] = False
                    if int(round(sum(np.abs(p_dep)), 4)) != 1:
                        message += f'\n[{exper}] WARNING sum of depolarizing probability of {edge} is not 1 to decimal 4.'
                        Parameters_test['depolarizing error'] = False

            # gate_error
            if gate_error > 1 or gate_error < 0:
                message += f'\n[{exper}] WARNING gate error probability is not a valid value.'
                Parameters_test['gate error'] = False

            # memory_error
            if callable(memory_error):
                for i in range(10):
                    x = random.random()
                    p = memory_error(x)
                    if len(p) != 4:
                        message += f'\n[{exper}] length of memory error is not 4'
                        Parameters_test['memory error'] = False
                        break
                    if int(round(sum(p), 2)) != 1:
                        message += f'\n[{exper}] WARNING sum of memory error probability of {edge} is not 1 to decimal 2.'
            else:
                message += f'\n[{exper}] WARNING memory error is not callable function'
                if len(memory_error) != 4:
                    message += f'\n[{exper}] length of memory error is not 4'
                    Parameters_test['memory error'] = False
            
            # measurement_error
            if measurement_error > 1:
                message += f'\n[{exper}] measurement error provided is more than 1'
                Parameters_test['measurement error'] = False

            # Valiate Process-flow and number of resource

            G = nx.Graph()
            for edge in topology:
                G.add_edge(edge[0], edge[1]) 
            nx.set_edge_attributes(G, topology)

            message += '\nLimited Process: '
            limited_processes = []
            for process in timeline:
                if type(process['Num Trials'] )is int:
                    limited_processes.append(process['Main Process'])

                    if tuple(process['Edges']) == (list(G.nodes())[0], list(G.nodes())[-1]):
                        is_end_to_end = f'End-to-end process -> {process["Edges"]}'
                    else:
                        is_end_to_end = f'Non-E2E process -> {process["Edges"]}'

                    message += f"\n{len(limited_processes)}. {process['Main Process']} : {is_end_to_end}"
            

            if len(limited_processes) == 0:
            # Detect no limited process simulation
                message += f'\n[{exper}] experiment has no limited process...'

                if sim_time is None:
                    message += f'\n[{exper}] WARNING: simulation might not terminate'
            
            if sim_time is not None:
                message += f'\n[{exper}] experiment will end with simulation time limit: {sim_time}s'

            if Dynamic_flag:
                message += f'\n[{exper}] experiment contains dynamic nodes...'
                if sim_time is None:
                    message += f'\n[{exper}] experiment will terminate only if limited process is finish, please make sure that loss and distance provided are reasonable as simulation might take insanely much time to finish...'

            if nx.is_connected(G):
                message += '\nTopology provied is connected graph...'
            else:
                message += '\nThe topology provided is not connected, this might not be what is expected...'

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
                message += '\nAll edges have Generate physical Bell pair process...'
            else:
                message += '\nNot all edge have Generate physical Bell pair process, the missing edges are, '
                for edge in missing_process_edge:
                    message += f'\n{edge}'

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
                                message += f"\n{e}"
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
                    message += f'\nLimited process: {limited_process} is reachable to fundamental resource processes...'
                    allGreen = True
                else:
                    message += f'\nLimited process: {limited_process} is not reachable to fundamental resource processes, this might cause an error'
                    allGreen = False

                if allGreen:
                    pass
                    # if purification check number of external qubits

                    # if logical encoding check number of internal encoding qubits

            if allGreen:
                exper_allGreen += 1
                message += f'\n[{exper}] all status checked '
            else:
                message += f'\nChecking is not pass, please be patient and re-check [{exper}] again.'


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
            if show_message:
                print(message)
            return dfStyler

        if exper_allGreen == len(self.strategies_list):
            message += '\nAll timeline and topology of all experiments are validated, you are good to execute Experiment.run() command!\
                 \nAnother error that is not currently check is the number of qubits needed to completed the task.'
            if show_message:
                print(message)
            return True
        else:
            message += '\nTest not passed.'
            if show_message:
                print(message)
            return False

    def execute(self, 
                multithreading:Optional[bool] = False, 
                save_result: Optional[bool] = False):
        """Execute experiments

        Args:
            multithreading (Optional[bool], optional): Whether to use multithreading or not. Defaults to False.
            save_result (Optional[bool], optional): Whether to save simulated result or not. Defaults to False.

        Returns:
            dict: result dictionary of simulated results.
        """
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

class Configuration:
    def __init__(self, 
                 edges_info: Dict, 
                 timeline: List, 
                 nodes_info: Optional[Dict] = None, 
                 memory_function: Optional[Union[Callable, List]] = None, 
                 gate_error: Optional[Union[int, float]] = None, 
                 measurement_error: Optional[Union[int, float]] = None, 
                 message:Optional[str] = None, 
                 throughtput_edges: Optional[List] = None, 
                 label_record: Optional[Any] = None, 
                 sim_time: Optional[Union[int, float]] = None, 
                 collectFidelityHistory: Optional[bool] = False, 
                 experiment: Optional[bool] = False, 
                 result_path: Optional[str] = 'result'):
        
        """Configuration instance to be feed to `QuantumNetwork`

        Args:
            edges_info (Dict): Each key is a tuple of nodes form edge.
            timeline (List): List of process in resource-process flow
            nodes_info (Optional[Dict], optional): Information of each node. Defaults to None.
            memory_function (Optional[Union[Callable, List]], optional): Memory function. Defaults to None.
            gate_error (Optional[Union[int, float]], optional): Probablity of gate error. Defaults to None.
            measurement_error (Optional[Union[int, float]], optional): Probablity of measurement error. Defaults to None.
            message (Optional[str], optional): Message to be added for saved simulated result. Defaults to None.
            throughtput_edges (Optional[List], optional): Edge to be used for connection setup. Defaults to None.
            label_record (Optional[Any], optional): Label of resource to be recorded. Defaults to None.
            sim_time (Optional[Union[int, float]], optional): Time in simulation allowed for simulation to execute. Defaults to None.
            collectFidelityHistory (Optional[bool], optional): Whether to collect fidelity for each measurement or not. Defaults to False.
            experiment (Optional[bool], optional): Addtion message to be used for saved result. Defaults to False.
            result_path (Optional[str], optional): Path to use for save simulated result. Defaults to 'result'.
        """

        self.numPhysicalBuffer = 20
        self.numInternalEncodingBuffer = 20
        self.numInternalDetectingBuffer = 10
        self.numInternalInterfaceBuffer = 2
        self.memFunc = np.inf if memory_function is None else memory_function # Function of memory of qubit
        self.gate_error = 0 if gate_error is None else gate_error
        self.measurement_error = 0 if measurement_error is None else measurement_error
        self.timeline = timeline
        self.experiment = experiment
        self.light_speed_in_fiber = 208189.206944 # km/s
        self.message = message
        self.g = edges_info
        self.result_path = result_path
        self.label_recorded = label_record
        self.collectFidelityHistory = collectFidelityHistory
        self.simulation_time = sim_time
        self.coor_system = 'normal'
        
        # Initialize graph
        G = nx.Graph()
        for edge in edges_info:

            for node in edge:

                if node not in edges_info[edge].keys():
                    edges_info[edge][node] = {
                        'memory function': self.memory_function,
                        'gate error': self.gate_error,
                        'measurement error': self.measurement_error
                    }

                if not callable(edges_info[edge][node]['memory function']):
                    memory_time = edges_info[edge][node]['memory function']
                    
                    # Memory error
                    def memory_error_function(time, tau=memory_time):
                        p = 0.75*np.e**(-1*(time/tau)) + 0.25
                        return [p, (1- p)/3, (1- p)/3, (1- p)/3]

                    edges_info[edge][node]['memory function'] = memory_error_function

            G.add_edge(edge[0], edge[1]) 

        nx.set_edge_attributes(G, edges_info)
        

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
                     _FidelityEstimation.Mixin,
                     _TimeLag.Mixin):

    def __init__(self, configuration: Configuration):
        """
        Quantum network instance initialize with configuration instance.
        The nessesary parameter and setting will be initialize for simulation.
        
            1. PhysicalQubit in each QNICs
            2. self.QubitTables which contain free qubit in each QNICs.
            3. self.resourceTables which contain entangle resource for each complete-edges.
            4. Parameter for network benchmarking.
            5. self.Expectation_value to be used for fidelity estimation.

        Args:
            configuration (Configuration): Configuration for simulation
        """

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
            for node1, node2, attr in self.graph.edges(data=True):  
                for i in range(self.table_name[table]):

                    self.QubitsTables[table][f'{node1}-{node2}'] \
                    [f'QNICs-{node1}'].put(PhysicalQubit(node1=node1, 
                                                         node2=node2, 
                                                         qubitID=i, 
                                                         qnic=f'{node1}-{node2}', 
                                                         role=table[:8], 
                                                         env=self.env, 
                                                         table=table, 
                                                         memFunc=attr[node1]['memory function'], 
                                                         gate_error=attr[node1]['gate error'], 
                                                         measurementError=attr[node1]['measurement error']))
                    self.QubitsTables[table][f'{node1}-{node2}'] \
                    [f'QNICs-{node2}'].put(PhysicalQubit(node1=node2, 
                                                         node2=node1, 
                                                         qubitID=i, 
                                                         qnic=f'{node1}-{node2}', 
                                                         role=table[:8], 
                                                         env=self.env, 
                                                         table=table, 
                                                         memFunc=attr[node2]['memory function'], 
                                                         gate_error=attr[node2]['gate error'], 
                                                         measurementError=attr[node2]['measurement error']))

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

        self.resourceTables = nx.complete_graph(self.graph.nodes)
        for node1, node2 in self.resourceTables.edges():
            for table in self.resource_table_name:
                if table[:8] == 'internal':
                    self.resourceTables[node1][node2][table] = {
                        f'{node1}' : simpy.FilterStore(self.env),
                        f'{node2}' : simpy.FilterStore(self.env),
                    }
                else:
                    self.resourceTables[node1][node2][table] = simpy.FilterStore(self.env)

        for process in self.configuration.timeline:
            process['isSuccess'] = 0

        self.simulationLog = [] 
        self.qubitsLog = []
        self.numResrouceProduced = {
            f'{node1}-{node2}' : {}
        for node1, node2 in self.complete_graph.edges()}
        self.numResrouceProduced = {}
        self.numBaseBellAttempt = 0
        self.numResourceUsedForFidelityEstimation = 0
        self.FidelityEstimationTimeStamp = None
        
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

    def createLinkResource(self, 
                           node1: str, 
                           node2: str, 
                           resource1: Union[PhysicalQubit, LogicalQubit], 
                           resource2: Union[PhysicalQubit, LogicalQubit],  
                           resource_table: Dict, 
                           label: Optional[str] = 'Physical'):
        """Method to put new entanglement into resource table

        Args:
            node1 (str): Node 1 containing resource 1
            node2 (str): Node 2 containing resource 2
            resource1 (Union[PhysicalQubit, LogicalQubit]): resource 1, one of the Bell pair
            resource2 (Union[PhysicalQubit, LogicalQubit]): resource 2, one of the Bell pair
            resource_table (Dict): Dictionary containing dict of each edge of simpy.FilterStore 
            label (Optional[str], optional): label of the new resource. Defaults to 'Physical'.

        Returns:
            None: 
        """
        resource1.isBusy, resource2.isBusy = True, True
        resource1.partner, resource2.partner = resource2, resource1
        resource1.partnerID, resource2.partnerID = resource2.qubitID, resource1.qubitID
        
        self.resourceTables[node1][node2][resource_table].put((resource1, resource2, label))
        # self.updateLog({'Time': self.env.now, 'Message': f'Qubit ({resource1.qubitID}) entangle with Qubit ({resource2.qubitID})'})

        if label in self.configuration.label_recorded:
            if f'{node1}-{node2}' not in self.numResrouceProduced:
                self.numResrouceProduced[f'{node1}-{node2}'] = {label: 1}
            else:
                if label not in self.numResrouceProduced[f'{node1}-{node2}']:
                    self.numResrouceProduced[f'{node1}-{node2}'][label] = 1
                else:
                    self.numResrouceProduced[f'{node1}-{node2}'][label] += 1

        # self.updateLog({'Time': self.env.now, 'Message': f'Qubit ({resource1.qubitID}) entangle with Qubit ({resource2.qubitID})'})

        return None

    def validateNodeOrder(self, 
                          node1: str, 
                          node2: str):
        """Method for arrange the right order of node which exist in 
           self.complete_edges_list which used to create self.resourceTables

        Args:
            node1 (str): Node 1.
            node2 (str): Node 2.

        Returns:
            (str, str): Validated node order.
        """

        if f'{node1}-{node2}' not in self.complete_edges_list:
            node1, node2 = node2, node1

        return node1, node2

    def Timeline(self):
        """
        Core process for simulation.
        The process consist of the following phase,
            1. Sending connection setup from self.configuration.throughtputEdges[0]
               to self.configuration.throughtputEdges[1] and back. The process needed to be finish before next phase.
            2. It will loop through each process in self.configuration.timeline to start each simulation.
               The processes will be divided into Limited_process and Unlimited_process. 
               This phase will finish only if Limited_process is finished.
            3. The self.connectionSetupTimeStamp will override with (self.env.now - itself).

        Raises:
            ValueError: Main process is not defined

        Yields:
            _type_: Successness of process
        """

        connectionSetup = [self.env.process(self.ConnectionSetup(self.configuration.throughtputEdges[0], 
                                                                 self.configuration.throughtputEdges[1]))]
        yield simpy.AllOf(self.env, connectionSetup)

        Unlimited_process = []
        Limited_process = []
        for process in self.configuration.timeline:
            
            if process['Main Process'] in ['PrototypeGeneratePhysicalResourcePulse', 'Generate physical Bell pair']: 
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
                p = [self.env.process(self.ExternalEntanglementSwapping(process,( process['Edges'][0], process['Edges'][1]), (process['Edges'][1], process['Edges'][2]), \
                                                                                num_required=process['Num Trials'], \
                                                                                label_in=process['Label in'], \
                                                                                label_out=process['Label out'], \
                                                                                resource_type=process['Resource Type'],
                                                                                note=process['Note']))]    
            elif process['Main Process'] in ['PrototypeStateTomography', 'State tomography', 'Fidelity estimation']:
                p = [self.env.process(self.FidelityEstimation(process, process['Edges'][0], process['Edges'][1], \
                                                                        num_required=process['Num Trials'], \
                                                                        label_in=process['Label in'], \
                                                                        resource_type=process['Resource Type'],
                                                                        note=process['Note']))] 
            elif process['Main Process'] in ['PrototypePurification', 'Entanglement purification', 'purification']:
                p = [self.env.process(self.Purification(process, process['Edges'][0], process['Edges'][1], 
                                                       num_required=process['Num Trials'], \
                                                       label_in=process['Label in'], \
                                                       label_out=process['Label out'], \
                                                       protocol=process['Protocol'],
                                                       note=process['Note']))]   
            elif process['Main Process'] in ['PrototypeGenerateLogicalResource', 'Generate logical Bell pair']:
                p = [self.env.process(self.GenerateLogicalResource(process, process['Edges'][0], process['Edges'][1], 
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

        self.connectionSetupTimeStamp = self.env.now - self.connectionSetupTimeStamp

    def run(self, 
            save_tomography: Optional[bool] = False, 
            save_result: Optional[bool] = True):
        """
        Execute the simulation and record the result

        Args:
            save_tomography (Optional[bool], optional): Whether to save separated file for state tomography or not. Defaults to False.
            save_result (Optional[bool], optional): Whether to save simulated result to file or not. Defaults to True.

        Returns:
            dict: result of simulation
        """

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
        config['Fidelity Estimation Time'] = self.FidelityEstimationTimeStamp
        config['Fidelity History'] = self.fidelityHistory
        config['Qubits waiting time'] = self.qubitsLog
        config['Commutation inforamtion'] = self.Expectation_value

        # Save log data
        config['Simulation log'] = self.simulationLog

        if save_result is True:
            with open(f"{self.configuration.result_path}/Result_{self.configuration.experiment}_{self.configuration.message}.pkl", "wb") as f:
                dill.dump(config, f)

        return config