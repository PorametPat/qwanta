import simpy 
import networkx as nx
import numpy as np
from typing import List, Union, Optional, Dict

class Mixin:
    def InternalEntanglementSwapping(self):
        pass

    def SimultanouseEntanglementSwapping(self):

        # TODO 
        '''
        Requirement 
        1. Implement classical message system with FilterStore
        2. Label unique id to each process (RuleSet id)
        3. Get qubit with correspond unique id inside the table (maybe new table)
        4. Perform Bell measurement and send result to end nodes

        OR (optional?)
        1. Implement classical message to assign qubit to particular implementation
        2. Separate qubit FilterStore for each node -> (qubit, entagle_with_node_[])
        '''
        pass

    # To let it independently process just have moniter while True moniter 
    # Bell pair and troll it to another processing process

    def ExternalEntanglementSwapping(self, 
                                              process: Dict, 
                                              edge1: List, 
                                              edge2: List, 
                                              num_required: Optional[int] = 1, 
                                              label_in: Optional[str] = 'Physical', 
                                              label_out: Optional[str] = 'Physical', 
                                              resource_type: Optional[str] = 'Physical', 
                                              note: Optional[Union[str, List]] = None):
        """This process will not induce any time delay, hence when `label_in` resources are available,
           it will fire an independent process for entanglement swapping which perform actual protocol.

        Args:
            process (Dict): Dictionary of contain information of process.
            edge1 (List): edge 1 which this process is process.
            edge2 (List): edge 2 which this process is process.
            num_required (Optional[int], optional): Number of time that this process needed to looped. Defaults to 1.
            label_in (Optional[str], optional): Input label of resource. Defaults to 'Physical'.
            label_out (Optional[str], optional): Output label of resource. Defaults to 'Purified'.
            resource_type (Optional[str], optional): Type of resource to be used in operation. Defaults to 'Ss-Dp'.
            note (Optional[Union[str, List]], optional): Addition note for process. Defaults to None.

        Yields:
            _type_: _description_
        """

        leftNode = edge1[0]; swapper = edge1[1]; rightNode = edge2[1]

        # Valiate node order
        leftNode, rightNode = self.validateNodeOrder(leftNode, rightNode)


        if resource_type == 'Physical':
            # table = self.resourceTables['physicalResourceTable']
            table = 'physicalResourceTable'
        elif resource_type == 'Logical':
            # table = self.resourceTables['logicalResourceTable']
            table = 'logicalResourceTable'
        
        while process['isSuccess'] < num_required: 

            # Just in case, validate order of node agian 
            # Case: (swapping-leftNode) (swapping-RightNode)
            tmp_left, tmp_swapping_left = self.validateNodeOrder(leftNode, swapper)
            tmp_swapping_right, tmp_right = self.validateNodeOrder(swapper, rightNode)

            if type(label_in) is str:
                label_in = [label_in]*2

            # get Bell pairs
            '''
            event = yield simpy.AllOf(self.env, [table[f'{tmp_left}-{tmp_swapping_left}'].get(lambda bell: bell[2]==label_in[0]), 
                                                 table[f'{tmp_swapping_right}-{tmp_right}'].get(lambda bell: bell[2]==label_in[1])])
            '''
            event = yield simpy.AllOf(self.env, [self.resourceTables[leftNode][swapper][table].get(lambda bell: bell[2]==label_in[0]), 
                                                 self.resourceTables[swapper][rightNode][table].get(lambda bell: bell[2]==label_in[1])])

            # Separate here?
            info = (event, leftNode, swapper, rightNode, table, resource_type, label_out, num_required, process, note)

            self.env.process(self._independentES(info))

    def _independentES(self, info):

        event, leftNode, swapper, rightNode, table, resource_type, label_out, num_required, process, note = info

        Bell_left = yield event.events[0]
        Bell_right = yield event.events[1]

        # Swapping (Bell_left[0]) ---- (Bell_left[1] , Bell_right[0]) ---- (Bell_right[1])
        new_Bell = (Bell_left[0], Bell_right[1])
        # Perform entanglement swapping here
        if resource_type == 'Logical':
            # error_detection_correction

            # Get internal qubit to encode on both side
            ancilla_qubits_left = [self.QubitsTables['internalDetectingQubitTable'][Bell_left[1].qnics_address] \
                                    [f'QNICs-{Bell_left[1].qubit_node_address}'].get() for _ in range(6) ]
            ancilla_qubits_right = [self.QubitsTables['internalDetectingQubitTable'][Bell_right[0].qnics_address] \
                                    [f'QNICs-{Bell_right[0].qubit_node_address}'].get() for _ in range(6) ]


            event = yield simpy.AllOf(self.env, [*ancilla_qubits_left, *ancilla_qubits_right])

            AncillaQubit_left = []
            for i in range(6):
                tmp = yield event.events[i]
                tmp.setInitialTime()
                AncillaQubit_left.append(tmp)

            Bell_left[1].ancilla_list = AncillaQubit_left

            AncillaQubit_right = []
            for i in range(6, 12):
                tmp = yield event.events[i]
                tmp.setInitialTime()
                AncillaQubit_right.append(tmp)

            Bell_right[0].ancilla_list = AncillaQubit_right

            perfect = True if note == 'Perfect' else False
            if type(perfect) != bool:
                raise ValueError('Note error')

            Bell_left[1].error_detection_correction(perfect_correction=perfect) 
            Bell_right[0].error_detection_correction(perfect_correction=perfect)

            Bell_right[0].CNOT_gate(Bell_left[1].physical_list, Bell_right[0].physical_list)

            Bell_left[1].error_detection_correction(perfect_correction=perfect) 
            Bell_right[0].error_detection_correction(perfect_correction=perfect)

            Bell_left[1].H_gate()
            Bell_left[1].error_detection_correction(perfect_correction=perfect) 

            # TODO Decode for measurement
            right_result = Bell_right[0].measure(basis='Z')
            left_result = Bell_left[1].measure(basis='Z')

            if type(right_result) != bool or type(left_result) != bool:
                raise ValueError('measure function of logical qubit return wrong type')

            Bell_left[1].setFree(); Bell_right[0].setFree()

            # Release physcial qubit for encoding
            for qu in Bell_left[1].physical_list:
                self.env.process(self.returnToQubitTable(qu))
            for qu in Bell_right[0].physical_list:
                self.env.process(self.returnToQubitTable(qu))

            # Release physcial qubit for detecting
            for qu in Bell_left[1].ancilla_list:
                qu.setFree()
                self.env.process(self.returnToQubitTable(qu))
            for qu in Bell_right[0].ancilla_list:
                qu.setFree()
                self.env.process(self.returnToQubitTable(qu))

        else:
            Bell_right[0].CNOT_gate(Bell_left[1])
            Bell_left[1].H_gate()
            
            right_result = Bell_right[0].measureZ()
            left_result = Bell_left[1].measureZ()

            if type(right_result) != bool or type(left_result) != bool:
                raise ValueError('measure function of physical qubit return wrong type')

            Bell_left[1].setFree(); Bell_right[0].setFree()
            
            self.env.process(self.returnToQubitTable(Bell_left[1]))
            self.env.process(self.returnToQubitTable(Bell_right[0]))

        # classical notification for result
        if nx.dijkstra_path_length(self.graph, leftNode, swapper) < nx.dijkstra_path_length(self.graph, rightNode, swapper):
            yield self.env.process(self.classicalCommunication(swapper, rightNode))
        else:
            yield self.env.process(self.classicalCommunication(swapper, leftNode))
        
        # Perfect error propagation.
        if left_result:
            Bell_right[1].Z_gate(gate_error=0)
        if right_result:
            Bell_right[1].X_gate(gate_error=0)
        
        # Apply error to qubit
        rand = np.random.random()
        if rand < 0.25:
            Bell_right[1].I_gate()
            Bell_right[1].I_gate()
        elif rand < 0.75:
            Bell_right[1].I_gate()
        else:
            pass
        
        self.createLinkResource(leftNode, rightNode, *new_Bell, table, label=label_out)

        if not isinstance(num_required, bool):
            process['isSuccess'] += 1