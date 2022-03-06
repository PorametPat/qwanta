import simpy 

class Mixin:
    def VirtualStateTomography(self, node1, node2, num_required=10000, label_in='Physical',resource_type='Physical', note=None):

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        if resource_type == 'Physical':
            table = self.resourceTables['physicalResourceTable']
        elif resource_type == 'Logical':
            table = self.resourceTables['logicalResourceTable']

        isSuccess = 0
        while isSuccess < num_required:

            Bell = yield table[f'{node1}-{node2}'].get(lambda bell: bell[2]==label_in)

            if Bell[0] == Bell[1]:
                raise ValueError('Qubits used for tomography are the same')

            if Bell[0].qubit_node_address != node1 or Bell[1].qubit_node_address != node2:
                raise ValueError('Qubits used for tomography are in the same address')
            
            if resource_type == 'Logical':
                # Get internal qubit to encode on both side
                ancilla_qubits_0 = [self.QubitsTables['internalDetectingQubitTable'][Bell[0].qnics_address] \
                                   [f'QNICs-{Bell[0].qubit_node_address}'].get() for _ in range(6) ]
                ancilla_qubits_1 = [self.QubitsTables['internalDetectingQubitTable'][Bell[1].qnics_address] \
                                   [f'QNICs-{Bell[1].qubit_node_address}'].get() for _ in range(6) ]

                event = yield simpy.AllOf(self.env, [*ancilla_qubits_0, *ancilla_qubits_1])

                #print(f'Logical Tomography at {node1}-{node2}')

                AncillaQubit_1 = []
                for i in range(6):
                    tmp = yield event.events[i]
                    tmp.setInitialTime()
                    AncillaQubit_1.append(tmp)

                Bell[0].ancilla_list = AncillaQubit_1

                AncillaQubit_2 = []
                for i in range(6, 12):
                    tmp = yield event.events[i]
                    tmp.setInitialTime()
                    AncillaQubit_2.append(tmp)

                Bell[1].ancilla_list = AncillaQubit_2

                Bell[0].error_detection_correction() 
                Bell[1].error_detection_correction()

                # Record wating time
                tmp1 = []; tmp2 = []
                for i in range(7):
                    tmp1.append(self.env.now - Bell[0].physical_list[i].initiateTime)
                    tmp2.append(self.env.now - Bell[1].physical_list[i].initiateTime)
                self.qubitsLog.append({f'{node1}': tmp1, f'{node2}': tmp2, 'Time': self.env.now})

                # Release ancilla qubit for detecting
                for i in range(len(Bell[0].ancilla_list)):
                    Bell[0].ancilla_list[i].setFree(); Bell[1].ancilla_list[i].setFree()
                    self.QubitsTables[Bell[0].ancilla_list[i].table][Bell[0].ancilla_list[i].qnics_address] \
                                     [f'QNICs-{Bell[0].ancilla_list[i].qubit_node_address}'].put(Bell[0].ancilla_list[i])
                    self.QubitsTables[Bell[1].ancilla_list[i].table][Bell[1].ancilla_list[i].qnics_address] \
                                     [f'QNICs-{Bell[1].ancilla_list[i].qubit_node_address}'].put(Bell[1].ancilla_list[i])

                for i in range(7):
                    #print(Bell[0].physical_list[i].qubitID, Bell[1].physical_list[i].qubitID)
                    if Bell[0].physical_list[i].initiateTime is None or Bell[1].physical_list[i].initiateTime is None:
                        raise ValueError("Initiate time is not set")

            # Stabilizer counting method
            result_1_SC = Bell[0].measureForFidelity(method='wow') # d='get_logical_operator')
            result_2_SC = Bell[1].measureForFidelity(method='wow') # method='get_logical_operator'   

            # State tomography
            result_1_ST = Bell[0].measureForFidelity(method='get_operator') # d='get_logical_operator')
            result_2_ST = Bell[1].measureForFidelity(method='get_operator') # method='get_logical_operator'          

            # Get error operator of each qubit 
            self.measurementResult.append({'qubit1': result_1_ST, 'qubit2': result_2_ST})

            # If the results are the same, it mean that error operator stabilize Bell state
            if result_1_SC == result_2_SC:
                self.stabilizerCount += 1
            self.measurementCount += 1

            if self.configuration.collectFidelityHistory:
                # Collect fidelity history
                self.fidelityHistory.append(self.stabilizerCount/self.measurementCount)
            
            if resource_type == 'Logical':
                Bell[0].setFree(); Bell[1].setFree()
                # Release physcial qubit for encoding
                for i in range(len(Bell[0].physical_list)):
                    self.QubitsTables[Bell[0].physical_list[i].table][Bell[0].physical_list[i].qnics_address] \
                                     [f'QNICs-{Bell[0].physical_list[i].qubit_node_address}'].put(Bell[0].physical_list[i])
                    self.QubitsTables[Bell[1].physical_list[i].table][Bell[1].physical_list[i].qnics_address] \
                                     [f'QNICs-{Bell[1].physical_list[i].qubit_node_address}'].put(Bell[1].physical_list[i])

            else:
                Bell[0].setFree(); Bell[1].setFree()
                self.QubitsTables[Bell[0].table][Bell[0].qnics_address][f'QNICs-{Bell[0].qubit_node_address}'].put(Bell[0])
                self.QubitsTables[Bell[1].table][Bell[1].qnics_address][f'QNICs-{Bell[1].qubit_node_address}'].put(Bell[1])
            
            # self.updateLog({'Time': self.env.now, 'Message': f'{resource_type} resource used'})
        
            if num_required is not True:
                isSuccess += 1


    def PrototypeVirtualStateTomography(self, process, node1, node2, num_required=10000, label_in='Physical',resource_type='Physical', note=None):

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        if resource_type == 'Physical':
            table = self.resourceTables['physicalResourceTable']
        elif resource_type == 'Logical':
            table = self.resourceTables['logicalResourceTable']
        
        num_measure_per_stab = int(num_required/3)

        while process['isSuccess'] < num_required:

            Bell = yield table[f'{node1}-{node2}'].get(lambda bell: bell[2]==label_in)

            if Bell[0] == Bell[1]:
                raise ValueError('Qubits used for tomography are the same')

            if Bell[0].qubit_node_address != node1 or Bell[1].qubit_node_address != node2:
                raise ValueError('Qubits used for tomography are in the same address')
            
            info = (Bell, node1, node2, resource_type, process, num_required, num_measure_per_stab, note)

            self.env.process(self._independentVirtualStateTomography(info))
            
    def _independentVirtualStateTomography(self, info):

        Bell, node1, node2, resource_type, process, num_required, num_measure_per_stab, note = info

        if resource_type == 'Logical':
            # Get internal qubit to encode on both side
            ancilla_qubits_0 = [self.QubitsTables['internalDetectingQubitTable'][Bell[0].qnics_address] \
                                [f'QNICs-{Bell[0].qubit_node_address}'].get() for _ in range(6) ]
            ancilla_qubits_1 = [self.QubitsTables['internalDetectingQubitTable'][Bell[1].qnics_address] \
                                [f'QNICs-{Bell[1].qubit_node_address}'].get() for _ in range(6) ]

            event = yield simpy.AllOf(self.env, [*ancilla_qubits_0, *ancilla_qubits_1])

            AncillaQubit_1 = []
            for i in range(6):
                tmp = yield event.events[i]
                tmp.setInitialTime()
                AncillaQubit_1.append(tmp)

            Bell[0].ancilla_list = AncillaQubit_1

            AncillaQubit_2 = []
            for i in range(6, 12):
                tmp = yield event.events[i]
                tmp.setInitialTime()
                AncillaQubit_2.append(tmp)

            Bell[1].ancilla_list = AncillaQubit_2

            perfect = True if note == 'Perfect' else False

            Bell[0].error_detection_correction(perfect_correction=perfect) 
            Bell[1].error_detection_correction(perfect_correction=perfect)

            # Record wating time
            tmp1 = []; tmp2 = []
            for i in range(7):
                tmp1.append(self.env.now - Bell[0].physical_list[i].initiateTime)
                tmp2.append(self.env.now - Bell[1].physical_list[i].initiateTime)
            self.qubitsLog.append({f'{node1}': tmp1, f'{node2}': tmp2, 'Time': self.env.now})

            # Release ancilla qubit for detecting
            for i in range(len(Bell[0].ancilla_list)):
                Bell[0].ancilla_list[i].setFree(); Bell[1].ancilla_list[i].setFree()
                self.QubitsTables[Bell[0].ancilla_list[i].table][Bell[0].ancilla_list[i].qnics_address] \
                                    [f'QNICs-{Bell[0].ancilla_list[i].qubit_node_address}'].put(Bell[0].ancilla_list[i])
                self.QubitsTables[Bell[1].ancilla_list[i].table][Bell[1].ancilla_list[i].qnics_address] \
                                    [f'QNICs-{Bell[1].ancilla_list[i].qubit_node_address}'].put(Bell[1].ancilla_list[i])

            for i in range(7):
                #print(Bell[0].physical_list[i].qubitID, Bell[1].physical_list[i].qubitID)
                if Bell[0].physical_list[i].initiateTime is None or Bell[1].physical_list[i].initiateTime is None:
                    raise ValueError("Initiate time is not set")
        else:
            tmp1 = [self.env.now - Bell[0].initiateTime]
            tmp2 = [self.env.now - Bell[1].initiateTime]
            self.qubitsLog.append({f'{node1}': tmp1, f'{node2}': tmp2, 'Time': self.env.now})


        # Stabilizer Measurement
        if process['isSuccess'] < num_measure_per_stab:
            # For stabilizer XX
            result_1, result_1_ST = Bell[0].measureX(get_operator=True)
            result_2, result_2_ST = Bell[1].measureX(get_operator=True)
            if result_1 == result_2:
                self.Expectation_value['XX']['commute'] += 1
            else:
                self.Expectation_value['XX']['anti-commute'] += 1

            # Get error operator of each qubit 
            self.measurementResult.append({'qubit1': result_1_ST, 'qubit2': result_2_ST})  
            

        elif process['isSuccess'] < 2*num_measure_per_stab:
            # For stabilizer YY
            result_1, result_1_ST = Bell[0].measureY(get_operator=True)
            result_2, result_2_ST = Bell[1].measureY(get_operator=True)
            if result_1 == result_2:
                self.Expectation_value['YY']['commute'] += 1
            else:
                self.Expectation_value['YY']['anti-commute'] += 1

            # Get error operator of each qubit 
            self.measurementResult.append({'qubit1': result_1_ST, 'qubit2': result_2_ST})  

        elif process['isSuccess'] < 3*num_measure_per_stab:
            # For stabilizer ZZ
            result_1, result_1_ST = Bell[0].measureZ(get_operator=True)
            result_2, result_2_ST = Bell[1].measureZ(get_operator=True)
            if result_1 == result_2:
                self.Expectation_value['ZZ']['commute'] += 1
            else:
                self.Expectation_value['ZZ']['anti-commute'] += 1

            # Get error operator of each qubit 
            self.measurementResult.append({'qubit1': result_1_ST, 'qubit2': result_2_ST})  
 
        
        if resource_type == 'Logical':
            Bell[0].setFree(); Bell[1].setFree()
            # Release physcial qubit for encoding
            for i in range(len(Bell[0].physical_list)):
                self.QubitsTables[Bell[0].physical_list[i].table][Bell[0].physical_list[i].qnics_address] \
                                    [f'QNICs-{Bell[0].physical_list[i].qubit_node_address}'].put(Bell[0].physical_list[i])
                self.QubitsTables[Bell[1].physical_list[i].table][Bell[1].physical_list[i].qnics_address] \
                                    [f'QNICs-{Bell[1].physical_list[i].qubit_node_address}'].put(Bell[1].physical_list[i])

        else:
            Bell[0].setFree(); Bell[1].setFree()
            self.QubitsTables[Bell[0].table][Bell[0].qnics_address][f'QNICs-{Bell[0].qubit_node_address}'].put(Bell[0])
            self.QubitsTables[Bell[1].table][Bell[1].qnics_address][f'QNICs-{Bell[1].qubit_node_address}'].put(Bell[1])
        
        # self.updateLog({'Time': self.env.now, 'Message': f'{resource_type} resource used'})
    
        if num_required is not True:
            process['isSuccess'] += 1

        if process['isSuccess'] == num_required:
            normalize_XX = (self.Expectation_value['XX']['commute'] - self.Expectation_value['XX']['anti-commute'])/(self.Expectation_value['XX']['commute'] + self.Expectation_value['XX']['anti-commute'])
            normalize_YY = (self.Expectation_value['YY']['commute'] - self.Expectation_value['YY']['anti-commute'])/(self.Expectation_value['YY']['commute'] + self.Expectation_value['YY']['anti-commute'])
            normalize_ZZ = (self.Expectation_value['ZZ']['commute'] - self.Expectation_value['ZZ']['anti-commute'])/(self.Expectation_value['ZZ']['commute'] + self.Expectation_value['ZZ']['anti-commute'])
            self.fidelityStabilizerMeasurement = (0.25)*(1 + normalize_XX + normalize_YY + normalize_ZZ)

        if self.configuration.collectFidelityHistory:
            # Collect fidelity history
            norm_XX = self.Expectation_value['XX']['commute'] + self.Expectation_value['XX']['anti-commute']
            norm_YY = self.Expectation_value['YY']['commute'] + self.Expectation_value['YY']['anti-commute']
            norm_ZZ = self.Expectation_value['ZZ']['commute'] + self.Expectation_value['ZZ']['anti-commute']
            if (norm_XX == 0) or (norm_YY == 0) or (norm_ZZ == 0):
                F = None
            else:
                normalize_XX = (self.Expectation_value['XX']['commute'] - self.Expectation_value['XX']['anti-commute'])/norm_XX
                normalize_YY = (self.Expectation_value['YY']['commute'] - self.Expectation_value['YY']['anti-commute'])/norm_YY
                normalize_ZZ = (self.Expectation_value['ZZ']['commute'] - self.Expectation_value['ZZ']['anti-commute'])/norm_ZZ
                F = (0.25)*(1 + normalize_XX + normalize_YY + normalize_ZZ)
            self.fidelityHistory.append(F)