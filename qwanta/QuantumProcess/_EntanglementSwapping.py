import simpy 
import networkx as nx
import numpy as np

class Mixin:
    def InternalEntanglementSwapping(self):
        pass

    def ExternalEntanglementSwapping(self, edge1, edge2, num_required=1, label_in='Physical', label_out='Physical', resource_type='Physical'):

        leftNode = edge1[0]; swapper = edge1[1]; rightNode = edge2[1]

        # Valiate node order
        leftNode, rightNode = self.validateNodeOrder(leftNode, rightNode)


        if resource_type == 'Physical':
            table = self.resourceTables['physicalResourceTable']
        elif resource_type == 'Logical':
            table = self.resourceTables['logicalResourceTable']
        
        isSuccess = 0
        while isSuccess < num_required: 

            # Just in case, validate order of node agian 
            # Case: (swapping-leftNode) (swapping-RightNode)
            tmp_left, tmp_swapping_left = self.validateNodeOrder(leftNode, swapper)
            tmp_swapping_right, tmp_right = self.validateNodeOrder(swapper, rightNode)

            if type(label_in) is str:
                label_in = [label_in]*2

            # get Bell pairs
            event = yield simpy.AllOf(self.env, [table[f'{tmp_left}-{tmp_swapping_left}'].get(lambda bell: bell[2]==label_in[0]), 
                                                 table[f'{tmp_swapping_right}-{tmp_right}'].get(lambda bell: bell[2]==label_in[1])])
    
            # Separate here?

            Bell_left = yield event.events[0]
            Bell_right = yield event.events[1]

            

            # Swapping (Bell_left[0]) ---- (Bell_left[1] , Bell_right[0]) ---- (Bell_right[1])
            new_Bell = (Bell_left[0], Bell_right[1])
            # Perform entanglement swapping here
            if resource_type == 'Logical':
                # error_detection_correction
                #print(f'Entanglement Swapping at {edge1}-{edge2}')

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

                Bell_left[1].error_detection_correction() 
                Bell_right[0].error_detection_correction()

                Bell_right[0].CNOT_gate(Bell_left[1].physical_list, Bell_right[0].physical_list)

                Bell_left[1].error_detection_correction() 
                Bell_right[0].error_detection_correction()

                Bell_left[1].H_gate()
                Bell_left[1].error_detection_correction() 

                Bell_left[1].setFree(); Bell_right[0].setFree()

                # TODO Decode for measurement?
                right_result = Bell_right[0].decode()
                left_result = Bell_left[1].decode()

                # Release physcial qubit for encoding
                for i in range(len(Bell_left[1].physical_list)):
                    self.QubitsTables[Bell_left[1].physical_list[i].table][Bell_left[1].physical_list[i].qnics_address] \
                                     [f'QNICs-{Bell_left[1].physical_list[i].qubit_node_address}'].put(Bell_left[1].physical_list[i])
                    self.QubitsTables[Bell_right[0].physical_list[i].table][Bell_right[0].physical_list[i].qnics_address] \
                                     [f'QNICs-{Bell_right[0].physical_list[i].qubit_node_address}'].put(Bell_right[0].physical_list[i])

                # Release physcial qubit for detecting
                for i in range(len(Bell_left[1].ancilla_list)):
                    Bell_left[1].ancilla_list[i].setFree();Bell_right[0].ancilla_list[i].setFree()
                    self.QubitsTables[Bell_left[1].ancilla_list[i].table][Bell_left[1].ancilla_list[i].qnics_address] \
                                     [f'QNICs-{Bell_left[1].ancilla_list[i].qubit_node_address}'].put(Bell_left[1].ancilla_list[i])
                    self.QubitsTables[Bell_right[0].ancilla_list[i].table][Bell_right[0].ancilla_list[i].qnics_address] \
                                     [f'QNICs-{Bell_right[0].ancilla_list[i].qubit_node_address}'].put(Bell_right[0].ancilla_list[i])

            else:
                Bell_right[0].CNOT_gate(Bell_left[1])
                Bell_left[1].H_gate()
                
                right_result = Bell_right[0].measureZ()
                left_result = Bell_left[1].measureZ()
 
                Bell_left[1].setFree(); Bell_right[0].setFree()

                self.QubitsTables[Bell_left[1].table][Bell_left[1].qnics_address][f'QNICs-{Bell_left[1].qubit_node_address}'].put(Bell_left[1])
                self.QubitsTables[Bell_right[0].table][Bell_right[0].qnics_address][f'QNICs-{Bell_right[0].qubit_node_address}'].put(Bell_right[0])

            # Error propagate via correction operation, to do the right wrong >~<
            if not left_result and right_result: # 0 1
                Bell_right[1].X_gate()
            elif left_result and not right_result: # 1 0 
                Bell_right[1].Z_gate()
            elif left_result and right_result: # 1 1
                Bell_right[1].Z_gate()
                Bell_right[1].X_gate()

            # classical notification for result
            if nx.dijkstra_path_length(self.graph, leftNode, swapper) < nx.dijkstra_path_length(self.graph, rightNode, swapper):
                yield self.env.process(self.classicalCommunication(swapper, rightNode))
            else:
                yield self.env.process(self.classicalCommunication(swapper, leftNode))

            self.updateLog({'Time': self.env.now, 'Message': f'Entanglement swapping for {leftNode}-{rightNode} success'})
            self.createLinkResource(leftNode, rightNode, *new_Bell, table, label=label_out)

            if not isinstance(num_required, bool):
                isSuccess += 1

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

    def PrototypeExternalEntanglementSwapping(self, process, edge1, edge2, num_required=1, label_in='Physical', label_out='Physical', resource_type='Physical', note=None):

        leftNode = edge1[0]; swapper = edge1[1]; rightNode = edge2[1]

        # Valiate node order
        leftNode, rightNode = self.validateNodeOrder(leftNode, rightNode)


        if resource_type == 'Physical':
            table = self.resourceTables['physicalResourceTable']
        elif resource_type == 'Logical':
            table = self.resourceTables['logicalResourceTable']
        
        while process['isSuccess'] < num_required: 

            # Just in case, validate order of node agian 
            # Case: (swapping-leftNode) (swapping-RightNode)
            tmp_left, tmp_swapping_left = self.validateNodeOrder(leftNode, swapper)
            tmp_swapping_right, tmp_right = self.validateNodeOrder(swapper, rightNode)

            if type(label_in) is str:
                label_in = [label_in]*2

            # get Bell pairs
            event = yield simpy.AllOf(self.env, [table[f'{tmp_left}-{tmp_swapping_left}'].get(lambda bell: bell[2]==label_in[0]), 
                                                 table[f'{tmp_swapping_right}-{tmp_right}'].get(lambda bell: bell[2]==label_in[1])])
    
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
            '''
            for i in range(len(Bell_left[1].physical_list)):
                self.QubitsTables[Bell_left[1].physical_list[i].table][Bell_left[1].physical_list[i].qnics_address] \
                                    [f'QNICs-{Bell_left[1].physical_list[i].qubit_node_address}'].put(Bell_left[1].physical_list[i])
                self.QubitsTables[Bell_right[0].physical_list[i].table][Bell_right[0].physical_list[i].qnics_address] \
                                    [f'QNICs-{Bell_right[0].physical_list[i].qubit_node_address}'].put(Bell_right[0].physical_list[i])
            '''
            for qu in Bell_left[1].physical_list:
                self.env.process(self.returnToQubitTable(qu))
            for qu in Bell_right[0].physical_list:
                self.env.process(self.returnToQubitTable(qu))

            # Release physcial qubit for detecting
            '''
            for i in range(len(Bell_left[1].ancilla_list)):
                Bell_left[1].ancilla_list[i].setFree();Bell_right[0].ancilla_list[i].setFree()
                self.QubitsTables[Bell_left[1].ancilla_list[i].table][Bell_left[1].ancilla_list[i].qnics_address] \
                                    [f'QNICs-{Bell_left[1].ancilla_list[i].qubit_node_address}'].put(Bell_left[1].ancilla_list[i])
                self.QubitsTables[Bell_right[0].ancilla_list[i].table][Bell_right[0].ancilla_list[i].qnics_address] \
                                    [f'QNICs-{Bell_right[0].ancilla_list[i].qubit_node_address}'].put(Bell_right[0].ancilla_list[i])
            '''
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

            #self.QubitsTables[Bell_left[1].table][Bell_left[1].qnics_address][f'QNICs-{Bell_left[1].qubit_node_address}'].put(Bell_left[1])
            #self.QubitsTables[Bell_right[0].table][Bell_right[0].qnics_address][f'QNICs-{Bell_right[0].qubit_node_address}'].put(Bell_right[0])
            
            self.env.process(self.returnToQubitTable(Bell_left[1]))
            self.env.process(self.returnToQubitTable(Bell_right[0]))

        # classical notification for result
        if nx.dijkstra_path_length(self.graph, leftNode, swapper) < nx.dijkstra_path_length(self.graph, rightNode, swapper):
            yield self.env.process(self.classicalCommunication(swapper, rightNode))
        else:
            yield self.env.process(self.classicalCommunication(swapper, leftNode))

        '''
        # Error propagate via correction operation, to do the right wrong >~<
        if not left_result and right_result: # 0 1
            Bell_right[1].X_gate()
        elif left_result and not right_result: # 1 0 
            Bell_right[1].Z_gate()
        elif left_result and right_result: # 1 1
            Bell_right[1].Z_gate()
            Bell_right[1].X_gate()
        '''
        
        # Or prehap? # Perfect error propagation.
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