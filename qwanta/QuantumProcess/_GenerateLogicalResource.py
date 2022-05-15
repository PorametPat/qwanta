import simpy 
import networkx as nx
import numpy as np
import uuid
from ..Qubit import LogicalQubit

class Mixin:
       
    def generateLogicalResource(self, node1, node2, num_required=1, label_in='Physical', label_out='Logical', protocol='Non-local CNOT'):

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        isSuccess = 0

        table = self.resourceTables['physicalResourceTable']
        result_table = self.resourceTables['logicalResourceTable']

        while isSuccess < num_required: 
                
            if protocol == 'Non-local CNOT':
                # Non-local CNOT style

                # Determine QNICs to use
                path = nx.dijkstra_path(self.configuration.NetworkTopology, node1, node2)

                # Get physical Bell pairs
                event_external = yield simpy.AllOf(self.env, [table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in) for _ in range(7)])
                Bells = []
                for bell in range(7):
                    tmp = yield event_external.events[bell]
                    Bells.append(tmp)
                if len(set(Bells)) != len(Bells):
                    raise ValueError('physical qubit that used to encode is not unique')

                # Get internal qubit to encode on both side
                node1_qubits = [self.QubitsTables['internalEncodingQubitTable'][f'{node1}-{path[1]}'][f'QNICs-{node1}'].get() for _ in range(7) ]
                node2_qubits = [self.QubitsTables['internalEncodingQubitTable'][f'{path[-2]}-{node2}'][f'QNICs-{node2}'].get() for _ in range(7) ]
                event_internal = yield simpy.AllOf(self.env, [*node1_qubits, *node2_qubits])

                # Separate here?
                #print(f'encode at {node1}-{node2}')
                self.updateLog({'Time': self.env.now, 'Message': f'Begin encoding logical Bell pair {node1}-{node2} using {label_in}'})

                # encode logical Bell pair
                physicalQubit_list1 = []
                for i in range(7):
                    tmp = yield event_internal.events[i]
                    tmp.setInitialTime()
                    physicalQubit_list1.append(tmp)

                id1 = uuid.uuid1()
                logicalQubit1 = LogicalQubit(node1, id1, f'{node1}-{path[1]}', self.env)
                logicalQubit1.physical_list = physicalQubit_list1
                logicalQubit1.encode()
                
                if len(set(logicalQubit1.physical_list)) != len(logicalQubit1.physical_list):
                    raise ValueError('physical qubit that used to encode is not unique')

                physicalQubit_list2 = []
                for i in range(7, 14):
                    tmp = yield event_internal.events[i]
                    tmp.setInitialTime()
                    physicalQubit_list2.append(tmp)

                id2 = uuid.uuid1()
                logicalQubit2 = LogicalQubit(node2, id2, f'{path[-2]}-{node2}', self.env)
                logicalQubit2.physical_list = physicalQubit_list2
                logicalQubit2.encode()
                
                if len(set(logicalQubit2.physical_list)) != len(logicalQubit2.physical_list):
                    raise ValueError('physical qubit that used to encode is not unique')

                

                # Perform non-local CNOT
                # Step 1: Transveral CNOT on left node
                for i in range(7):
                    if Bells[i][0].qubit_node_address != node1 or Bells[i][1].qubit_node_address != node2:
                        raise ValueError('Physical Address is not match')
                    if Bells[i][0].initiateTime is None or Bells[i][1].initiateTime is None:
                        raise ValueError('Initiate time is not set')
                    Bells[i][0].CNOT_gate(logicalQubit1.physical_list[i])
                
                # Measure and propagate error to remaining qubits via correction operation, [may combine with above process?]
                results = []
                for i in range(7):
                    results.append(Bells[i][0].measureZ())
                
                # Sent measurement result to another node
                yield self.env.process(self.classicalCommunication(node1, node2))
                for result in results:
                    if result:
                        Bells[i][1].X_gate()
                
                # Step 2
                for i in range(7):
                    logicalQubit2.physical_list[i].CNOT_gate(Bells[i][1])
                    Bells[i][1].H_gate()
                
                # measure
                results = []
                for i in range(7):
                    results.append(Bells[i][1].measureZ())

                # Sent measurement result to another node
                yield self.env.process(self.classicalCommunication(node2, node1))
                for result in results:
                    if result:
                        logicalQubit1.physical_list[i].Z_gate()

                # Set Bell pair free
                for bell in Bells:
                    bell[0].setFree(); bell[1].setFree()
                    # Feed qubit back to external resource
                    self.QubitsTables[bell[0].table][bell[0].qnics_address][f'QNICs-{bell[0].qubit_node_address}'].put(bell[0])
                    self.QubitsTables[bell[1].table][bell[1].qnics_address][f'QNICs-{bell[1].qubit_node_address}'].put(bell[1])
                
                # Add new logical resource 
                self.createLinkResource(node1, node2, logicalQubit1, logicalQubit2, result_table, label_out)

            elif protocol == 'Purified-encoded':
                # Determine QNICs to use
                path = nx.dijkstra_path(self.configuration.NetworkTopology, node1, node2)

                # Get physical Bell pairs
                event_external = yield simpy.AllOf(self.env, [table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in) for _ in range(1)])

                # Get internal qubit to encode on both side
                node1_qubits = [self.QubitsTables['internalEncodingQubitTable'][f'{node1}-{path[1]}'][f'QNICs-{node1}'].get() for _ in range(6) ]
                node2_qubits = [self.QubitsTables['internalEncodingQubitTable'][f'{path[-2]}-{node2}'][f'QNICs-{node2}'].get() for _ in range(6) ]
                event_internal = yield simpy.AllOf(self.env, [*node1_qubits, *node2_qubits])


                # encode logical Bell pair
                encode_qubit1 = yield event_external.events[0]
                physicalQubit_list1 = [encode_qubit1[0]]
                for i in range(6):
                    tmp = yield event_internal.events[i]
                    tmp.setInitialTime()
                    physicalQubit_list1.append(tmp)

                id1 = uuid.uuid1()
                logicalQubit1 = LogicalQubit(node1, id1, f'{node1}-{path[1]}', self.env)
                logicalQubit1.physical_list = physicalQubit_list1
                logicalQubit1.encode()
                
                if len(set(logicalQubit1.physical_list)) != len(logicalQubit1.physical_list):
                    raise ValueError('physical qubit that used to encode is not unique')

                encode_qubit2 = yield event_external.events[0]
                physicalQubit_list2 = [encode_qubit2[1]]
                for i in range(6, 12):
                    tmp = yield event_internal.events[i]
                    tmp.setInitialTime()
                    physicalQubit_list2.append(tmp)

                id2 = uuid.uuid1()
                logicalQubit2 = LogicalQubit(node2, id2, f'{path[-2]}-{node2}', self.env)
                logicalQubit2.physical_list = physicalQubit_list2
                logicalQubit2.encode()
                
                if len(set(logicalQubit2.physical_list)) != len(logicalQubit2.physical_list):
                    raise ValueError('physical qubit that used to encode is not unique')

                for i in range(7):
                    if logicalQubit2.physical_list[i].initiateTime is None or logicalQubit1.physical_list[i].initiateTime is None:
                        raise ValueError('Initiate time is not set.')
           
                self.createLinkResource(node1, node2, logicalQubit1, logicalQubit2, result_table, label_out)

            elif protocol == 'Physical-encoded':
                pass
            else:
                raise ValueError(f'Encoding protocol is {protocol} is not defined.')

            if not isinstance(num_required, bool):
                isSuccess += 1


    def PrototypeGenerateLogicalResource(self, process, node1, node2, num_required=1, label_in='Physical', label_out='Logical', protocol='Non-local CNOT'):

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        table = self.resourceTables['physicalResourceTable']
        result_table = self.resourceTables['logicalResourceTable']

        while process['isSuccess'] < num_required: 
                
            if protocol == 'Non-local CNOT':
                # Non-local CNOT style

                # Determine QNICs to use
                # path = nx.dijkstra_path(self.configuration.NetworkTopology, node1, node2)
                
                # Get physical Bell pairs
                event_external = yield simpy.AllOf(self.env, [table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in) for _ in range(7)])
                Bells = []
                for bell in range(7):
                    tmp = yield event_external.events[bell]
                    Bells.append(tmp)
                if len(set(Bells)) != len(Bells):
                    raise ValueError('physical qubit that used to encode is not unique')

                Qubit_node1_QNICs_address = Bells[0][0].qnics_address
                Qubit_node2_QNICs_address = Bells[0][1].qnics_address

                # Get internal qubit to encode on both side
                node1_qubits = [self.QubitsTables['internalEncodingQubitTable'][Qubit_node1_QNICs_address][f'QNICs-{node1}'].get() for _ in range(7) ]
                node2_qubits = [self.QubitsTables['internalEncodingQubitTable'][Qubit_node2_QNICs_address][f'QNICs-{node2}'].get() for _ in range(7) ]
                event_internal = yield simpy.AllOf(self.env, [*node1_qubits, *node2_qubits])

                info = (event_internal, Bells, node1, node2, Qubit_node1_QNICs_address , Qubit_node2_QNICs_address, table, result_table, label_out, num_required, process)
                self.env.process(self._independentNonLocalCNOT(info))

            elif protocol == 'Purified-encoded':
                # Determine QNICs to use
                # path = nx.dijkstra_path(self.configuration.NetworkTopology, node1, node2)
                
                # Get physical Bell pairs
                event_external = yield simpy.AllOf(self.env, [table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in) for _ in range(1)])
                encode_qubits = yield event_external.events[0]

                encode_qubit1 = encode_qubits[0]
                encode_qubit2 = encode_qubits[1]

                Qubit_node1_QNICs_address = encode_qubit1.qnics_address
                Qubit_node2_QNICs_address = encode_qubit2.qnics_address

                # Get internal qubit to encode on both side
                node1_qubits = [self.QubitsTables['internalEncodingQubitTable'][Qubit_node1_QNICs_address][f'QNICs-{node1}'].get() for _ in range(6) ]
                node2_qubits = [self.QubitsTables['internalEncodingQubitTable'][Qubit_node2_QNICs_address][f'QNICs-{node2}'].get() for _ in range(6) ]
                event_internal = yield simpy.AllOf(self.env, [*node1_qubits, *node2_qubits])

                info = (event_internal, encode_qubit1, encode_qubit2, node1, node2, Qubit_node1_QNICs_address , Qubit_node2_QNICs_address, result_table, label_out, num_required, process)
                self.env.process(self._independentPurifiedEncoded(info))

            elif protocol == 'Physical-encoded':
                pass


    def _independentNonLocalCNOT(self, info):

        # Non-local CNOT style

        event_internal, Bells,node1, node2, Qubit_node1_QNICs_address, Qubit_node2_QNICs_address, table, result_table, label_out, num_required, process = info

        # encode logical Bell pair
        physicalQubit_list1 = []
        for i in range(7):
            tmp = yield event_internal.events[i]
            tmp.setInitialTime()
            physicalQubit_list1.append(tmp)

        id1 = uuid.uuid1()
        logicalQubit1 = LogicalQubit(node1, id1, Qubit_node1_QNICs_address, self.env)
        logicalQubit1.physical_list = physicalQubit_list1
        logicalQubit1.encode()
        
        if len(set(logicalQubit1.physical_list)) != len(logicalQubit1.physical_list):
            raise ValueError('physical qubit that used to encode is not unique')

        physicalQubit_list2 = []
        for i in range(7, 14):
            tmp = yield event_internal.events[i]
            tmp.setInitialTime()
            physicalQubit_list2.append(tmp)

        id2 = uuid.uuid1()
        logicalQubit2 = LogicalQubit(node2, id2, Qubit_node2_QNICs_address, self.env)
        logicalQubit2.physical_list = physicalQubit_list2
        logicalQubit2.encode()
        
        if len(set(logicalQubit2.physical_list)) != len(logicalQubit2.physical_list):
            raise ValueError('physical qubit that used to encode is not unique')

        # Perform non-local CNOT
        # Step 1: Transveral CNOT on left node
        for i in range(7):
            if Bells[i][0].qubit_node_address != node1 or Bells[i][1].qubit_node_address != node2:
                raise ValueError('Physical Address is not match')
            if Bells[i][0].initiateTime is None or Bells[i][1].initiateTime is None:
                raise ValueError('Initiate time is not set')
            Bells[i][0].CNOT_gate(logicalQubit1.physical_list[i])
        
        # Measure and propagate error to remaining qubits via correction operation, [may combine with above process?]
        results = []
        for i in range(7):
            results.append(Bells[i][0].measureZ())
        
        # Sent measurement result to another node
        yield self.env.process(self.classicalCommunication(node1, node2))
        for result in results:
            if result:
                Bells[i][1].X_gate(gate_error=0)
            
            x = np.random.random()
            if x > 0.5:
                Bells[i][1].I_gate()
        
        # Step 2
        for i in range(7):
            logicalQubit2.physical_list[i].CNOT_gate(Bells[i][1])
            Bells[i][1].H_gate()
        
        # measure
        results = []
        for i in range(7):
            results.append(Bells[i][1].measureZ())

        # Sent measurement result to another node
        yield self.env.process(self.classicalCommunication(node2, node1))
        for result in results:
            if result:
                logicalQubit1.physical_list[i].Z_gate(gate_error=0)
            
            x = np.random.random()
            if x > 0.5:
                logicalQubit1.physical_list[i].I_gate()

        # Set Bell pair free
        for bell in Bells:
            bell[0].setFree(); bell[1].setFree()
            # Feed qubit back to external resource
            # self.QubitsTables[bell[0].table][bell[0].qnics_address][f'QNICs-{bell[0].qubit_node_address}'].put(bell[0])
            # self.QubitsTables[bell[1].table][bell[1].qnics_address][f'QNICs-{bell[1].qubit_node_address}'].put(bell[1])

            self.env.process(self.returnToQubitTable(bell[0]))
            self.env.process(self.returnToQubitTable(bell[1]))
        
        # Add new logical resource 
        self.createLinkResource(node1, node2, logicalQubit1, logicalQubit2, result_table, label_out)

        if not isinstance(num_required, bool):
            process['isSuccess'] += 1

    def _independentPurifiedEncoded(self, info):

        # Directly encoded logical Bell pair
        event_internal, encode_qubit1, encode_qubit2, node1, node2, Qubit_node1_QNICs_address , Qubit_node2_QNICs_address, result_table, label_out, num_required, process = info
        
        # encode logical Bell pair
        # encode_qubit1 = yield event_external.events[0]
        physicalQubit_list1 = [encode_qubit1]
        for i in range(6):
            tmp = yield event_internal.events[i]
            tmp.setInitialTime()
            physicalQubit_list1.append(tmp)

        id1 = uuid.uuid1()
        logicalQubit1 = LogicalQubit(node1, id1, Qubit_node1_QNICs_address, self.env)
        logicalQubit1.physical_list = physicalQubit_list1
        logicalQubit1.encode()
        
        if len(set(logicalQubit1.physical_list)) != len(logicalQubit1.physical_list):
            raise ValueError('physical qubit that used to encode is not unique')

        # encode_qubit2 = yield event_external.events[0]
        physicalQubit_list2 = [encode_qubit2]
        for i in range(6, 12):
            tmp = yield event_internal.events[i]
            tmp.setInitialTime()
            physicalQubit_list2.append(tmp)

        id2 = uuid.uuid1()
        logicalQubit2 = LogicalQubit(node2, id2, Qubit_node2_QNICs_address, self.env)
        logicalQubit2.physical_list = physicalQubit_list2
        logicalQubit2.encode()
        
        if len(set(logicalQubit2.physical_list)) != len(logicalQubit2.physical_list):
            raise ValueError('physical qubit that used to encode is not unique')

        for i in range(7):
            if logicalQubit2.physical_list[i].initiateTime is None or logicalQubit1.physical_list[i].initiateTime is None:
                raise ValueError('Initiate time is not set.')
    
        self.createLinkResource(node1, node2, logicalQubit1, logicalQubit2, result_table, label_out)

        if not isinstance(num_required, bool):
            process['isSuccess'] += 1