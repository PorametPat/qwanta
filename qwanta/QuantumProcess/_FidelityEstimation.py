from optparse import Option
import simpy 
from typing import Dict, Any, Optional, Union, List

class Mixin:

    def FidelityEstimation(self, 
                           process: Dict, 
                           node1: Any, 
                           node2: Any, 
                           num_required: Optional[int]=10000, 
                           label_in: Optional[str] = 'Physical',
                           resource_type: Optional[str] = 'Physical', 
                           note: Optional[Union[str, List]] = None):
        """This process will not induce any time delay, hence when `label_in` resources are available,
           it will fire an independent process for fidelity estimation which perform actual protocol.

        Args:
            process (Dict): Dictionary of contain information of process.
            node1 (Any): node 1 which this process is process.
            node2 (Any): node 2 which this process is process.
            num_required (Optional[int], optional): Number of time that this process needed to looped. Defaults to 10000.
            label_in (Optional[str], optional): Input label of resource. Defaults to 'Physical'.
            resource_type (Optional[str], optional): Type of resource to be used in operation. Defaults to 'Physical'.
            note (Optional[Union[str, List]], optional): Addition note for process. Defaults to None.

        Raises:
            ValueError: Qubits used for fidelity estimation are the same
            ValueError: Qubits used for fidelity estimation are in the same address

        Yields:
            _type_: _description_
        """

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
                raise ValueError('Qubits used for fidelity estimation are the same')

            if Bell[0].qubit_node_address != node1 or Bell[1].qubit_node_address != node2:
                raise ValueError('Qubits used for fidelity estimation are in the same address')
            
            info = (Bell, node1, node2, resource_type, process, num_required, num_measure_per_stab, note)

            self.env.process(self._independentFidelityEstimation(info))
            
    def _independentFidelityEstimation(self, info):

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

            for qu in Bell[1].ancilla_list:
                qu.setFree()
                self.env.process(self.returnToQubitTable(qu))
            for qu in Bell[0].ancilla_list:
                qu.setFree()
                self.env.process(self.returnToQubitTable(qu))

            for i in range(7):

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

            self.numResourceUsedForFidelityEstimation += 1
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

            self.numResourceUsedForFidelityEstimation += 1
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

            self.numResourceUsedForFidelityEstimation += 1
            # Get error operator of each qubit 
            self.measurementResult.append({'qubit1': result_1_ST, 'qubit2': result_2_ST}) 

            # Record final time stamp
            if (self.numResourceUsedForFidelityEstimation == int(3*num_measure_per_stab)):
                self.FidelityEstimationTimeStamp = self.env.now - self.FidelityEstimationTimeStamp 
 
        
        if resource_type == 'Logical':
            Bell[0].setFree(); Bell[1].setFree()
            # Release physcial qubit for encoding

            for qu in Bell[0].physical_list:
                self.env.process(self.returnToQubitTable(qu))
            for qu in Bell[1].physical_list:
                self.env.process(self.returnToQubitTable(qu))

        else:
            Bell[0].setFree(); Bell[1].setFree()

            self.env.process(self.returnToQubitTable(Bell[0]))
            self.env.process(self.returnToQubitTable(Bell[1]))

        # self.updateLog({'Time': self.env.now, 'Message': f'{resource_type} resource used'})
    
        if not isinstance(num_required, bool):
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