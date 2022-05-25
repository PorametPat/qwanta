import simpy 
import networkx as nx
import numpy as np
import uuid
from ..Qubit import LogicalQubit
from typing import Any, Optional, Dict

class Mixin:

    def GenerateLogicalResource(self, 
                                         process: Dict, 
                                         node1: Any, 
                                         node2: Any, 
                                         num_required: Optional[int] = 1, 
                                         label_in: Optional[str] = 'Physical', 
                                         label_out: Optional[str] = 'Logical', 
                                         protocol: Optional[str] = 'Non-local CNOT'):
        """This process will not induce any time delay, hence when `label_in` resources are available,
           it will fire an independent process for producing logical Bell pair which perform actual protocol.

        Args:
            process (Dict): Dictionary of contain information of process.
            node1 (Any): node 1 which this process is process.
            node2 (Any): node 2 which this process is process.
            num_required (Optional[int], optional): Number of time that this process needed to looped. Defaults to 1.
            label_in (Optional[str], optional): Input label of resource. Defaults to 'Physical'.
            label_out (Optional[str], optional): Output label of resource. Defaults to 'Logical'.
            protocol (Optional[str], optional): Protocol used for generation of logical Bell pair. Defaults to 'Non-local CNOT'.

        Raises:
            ValueError: Physical qubits that used to encode is not unique.

        Yields:
            _type_: _description_
        """

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        #table = self.resourceTables['physicalResourceTable']
        #result_table = self.resourceTables['logicalResourceTable']

        table = 'physicalResourceTable'
        result_table = 'logicalResourceTable'

        while process['isSuccess'] < num_required: 
                
            if protocol == 'Non-local CNOT':
                # Non-local CNOT style

                # Get physical Bell pairs
                # event_external = yield simpy.AllOf(self.env, [table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in) for _ in range(7)])
                event_external = yield simpy.AllOf(self.env, [self.resourceTables[node1][node2][table].get(lambda bell: bell[2] == label_in) for _ in range(7)])
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

                # Get physical Bell pairs
                event_external = yield simpy.AllOf(self.env, [self.resourceTables[node1][node2][table].get(lambda bell: bell[2] == label_in) for _ in range(1)])
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
        for index, result in enumerate(results):
            if result:
                Bells[index][1].X_gate(gate_error=0)
            
            x = np.random.random()
            if x > 0.5:
                Bells[index][1].I_gate()
        
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
        for index, result in enumerate(results):
            if result:
                logicalQubit1.physical_list[index].Z_gate(gate_error=0)
            
            x = np.random.random()
            if x > 0.5:
                logicalQubit1.physical_list[index].I_gate()

        # Set Bell pair free
        for bell in Bells:
            bell[0].setFree(); bell[1].setFree()
            # Feed qubit back to external resource

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