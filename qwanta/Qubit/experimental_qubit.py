import numpy as np 
import random

class PerfectCodeLogicalQubit:

    def __init__(self, node1, qubitID, qnic, env, code='5QPC'):
        # Simulation setup
        self.env = env

        # Network setup
        self.qubitID = f'{node1}-{qubitID}'
        self.qnics_address = qnic
        self.qubit_node_address = node1

        # Physical List
        self.code = code
        self.physical_list = []
        self.ancilla_list = []
        self.entangle_partner = []

    def setFree(self):
        
        for qubit in self.physical_list:
            qubit.setFree()

        return None

    def I_gate(self, gate_error=None):
        
        for i in self.physical_list:
            i.I_gate(gate_error=gate_error)

        return None

    def H_gate(self, gate_error=None):

        for i in self.physical_list:
            i.H_gate(gate_error=gate_error)

        # Reindex 
        self.physical_list = [self.physical_list[i] for i in [1, 4, 2, 0, 3]]

        return None

    def X_gate(self, gate_error=None):

        for i in self.physical_list:
            i.X_gate(gate_error=gate_error)

        return None

    def Z_gate(self, gate_error=None):

        for i in self.physical_list:
            i.Z_gate(gate_error=gate_error)

        return None

    def S_dagger_gate(self, gate_error=None):

        for i in self.physical_list:
            i.S_dagger_gate(gate_error=gate_error)

        return None

    def S_gate(self, gate_error=None):

        for i in self.physical_list:
            i.S_gate(gate_error=gate_error)

        return None

    def CNOT_gate(self, control_qubits, target_qubits, gate_error=None):

        for i in range(len(control_qubits)):
            target_qubits[i].CNOT_gate(control_qubits[i], gate_error=gate_error)

        return None

    def encode(self, apporach='standard'):

        if apporach=='standard':

            self.physical_list[0].Z_gate()
            self.physical_list[2].H_gate()
            self.physical_list[3].H_gate()

            self.physical_list[0].S_dagger_gate()
            self.physical_list[1].CNOT_gate(self.physical_list[3])
            
            self.physical_list[1].H_gate()
            self.physical_list[4].CNOT_gate(self.physical_list[2])

            self.physical_list[0].CNOT_gate(self.physical_list[1])
            self.physical_list[2].S_dagger_gate()
            self.physical_list[4].CNOT_gate(self.physical_list[3])

            self.physical_list[0].S_gate()
            self.physical_list[1].S_gate()
            self.physical_list[2].Z_gate()
            self.physical_list[3].S_gate()
            self.physical_list[4].S_dagger_gate()

            self.physical_list[0].CNOT_gate(self.physical_list[4])
            self.physical_list[4].H_gate()
            self.physical_list[1].CNOT_gate(self.physical_list[4])
        
        elif apporach == 'FFT':

            for i in self.physical_list:
                i.H_gate()

            self.physical_list[1].CZ_gate(self.physical_list[0])
            self.physical_list[3].CZ_gate(self.physical_list[2])
            self.physical_list[2].CZ_gate(self.physical_list[1])
            self.physical_list[4].CZ_gate(self.physical_list[3])
            self.physical_list[0].CZ_gate(self.physical_list[4])

        return None


    def decode(self, apporach='standard', no_error=False):

        gate_error = 0 if no_error else None

        if apporach=='standard':
           
            self.physical_list[1].CNOT_gate(self.physical_list[4])
            self.physical_list[4].H_gate()
            self.physical_list[0].CNOT_gate(self.physical_list[4])

            self.physical_list[4].S_dagger_gate()
            self.physical_list[3].S_gate()
            self.physical_list[2].Z_gate()
            self.physical_list[1].S_gate()
            self.physical_list[0].S_gate()


            self.physical_list[4].CNOT_gate(self.physical_list[3])
            self.physical_list[2].S_dagger_gate()
            self.physical_list[0].CNOT_gate(self.physical_list[1])
            
            self.physical_list[4].CNOT_gate(self.physical_list[2])
            self.physical_list[1].H_gate()

            self.physical_list[1].CNOT_gate(self.physical_list[3])
            self.physical_list[0].S_dagger_gate()

            self.physical_list[3].H_gate()
            self.physical_list[2].H_gate()
            self.physical_list[0].Z_gate()

        elif apporach == 'FFT':

            self.physical_list[0].CZ_gate(self.physical_list[4])
            self.physical_list[4].CZ_gate(self.physical_list[3])
            self.physical_list[2].CZ_gate(self.physical_list[1])
            self.physical_list[3].CZ_gate(self.physical_list[2])
            self.physical_list[1].CZ_gate(self.physical_list[0])
            
            for i in self.physical_list:
                i.H_gate()

        return None

class LogicalQubit:

    def __init__(self, node1, qubitID, qnic, env, code='Steane'):
        # Simulation setup
        self.env = env

        # Network setup
        self.qubitID = f'{node1}-{qubitID}'
        self.qnics_address = qnic
        self.qubit_node_address = node1

        # Physical List
        self.code = code
        self.physical_list = []
        self.ancilla_list = []
        self.entangle_partner = []

    def setFree(self):
        
        for qubit in self.physical_list:
            qubit.setFree()

        return None

    def I_gate(self, gate_error=None):
        
        if self.code == 'Steane':
            for i in self.physical_list:
                i.I_gate(gate_error=gate_error)

        return None

    def H_gate(self, gate_error=None):

        if self.code == 'Steane':
            for i in self.physical_list:
                i.H_gate(gate_error=gate_error)

        return None

    def X_gate(self, gate_error=None):

        if self.code == 'Steane':
            for i in self.physical_list:
                i.X_gate(gate_error=gate_error)

        return None

    def Z_gate(self, gate_error=None):

        if self.code == 'Steane':
            for i in self.physical_list:
                i.Z_gate(gate_error=gate_error)

        return None

    def S_dagger_gate(self, gate_error=None):

        if self.code == 'Steane':
            for i in self.physical_list:
                i.S_dagger_gate(gate_error=gate_error)

        return None

    def S_gate(self, gate_error=None):

        if self.code == 'Steane':
            for i in self.physical_list:
                i.S_gate(gate_error=gate_error)

        return None

    def CNOT_gate(self, control_qubits, target_qubits, gate_error=None):

        if self.code == 'Steane':
            for i in range(len(control_qubits)):
                target_qubits[i].CNOT_gate(control_qubits[i], gate_error=gate_error)

        return None

    def encode(self):

        if self.code == 'Steane':
            # Swap input to index 2
            self.physical_list[0], self.physical_list[2] = self.physical_list[2], self.physical_list[0]

            # Hard code
            self.physical_list[0].H_gate()
            self.physical_list[1].H_gate()
            self.physical_list[3].H_gate()

            self.physical_list[4].CNOT_gate(self.physical_list[2])
            self.physical_list[5].CNOT_gate(self.physical_list[2])

            self.physical_list[2].CNOT_gate(self.physical_list[0])
            self.physical_list[4].CNOT_gate(self.physical_list[0])
            self.physical_list[6].CNOT_gate(self.physical_list[0])

            self.physical_list[2].CNOT_gate(self.physical_list[1])
            self.physical_list[5].CNOT_gate(self.physical_list[1])
            self.physical_list[6].CNOT_gate(self.physical_list[1])

            self.physical_list[4].CNOT_gate(self.physical_list[3])
            self.physical_list[5].CNOT_gate(self.physical_list[3])
            self.physical_list[6].CNOT_gate(self.physical_list[3])

        return None


    def decode(self, no_error=False):

        gate_error = 0 if no_error else None

        if self.code == 'Steane':
           
            self.physical_list[6].CNOT_gate(self.physical_list[3], gate_error)
            self.physical_list[5].CNOT_gate(self.physical_list[3], gate_error)
            self.physical_list[4].CNOT_gate(self.physical_list[3], gate_error)

            self.physical_list[6].CNOT_gate(self.physical_list[1], gate_error)  
            self.physical_list[5].CNOT_gate(self.physical_list[1], gate_error) 
            self.physical_list[2].CNOT_gate(self.physical_list[1], gate_error)

            self.physical_list[6].CNOT_gate(self.physical_list[0], gate_error)
            self.physical_list[4].CNOT_gate(self.physical_list[0], gate_error)
            self.physical_list[2].CNOT_gate(self.physical_list[0], gate_error)

            self.physical_list[5].CNOT_gate(self.physical_list[2], gate_error)
            self.physical_list[4].CNOT_gate(self.physical_list[2], gate_error)

            self.physical_list[3].H_gate(gate_error)
            self.physical_list[1].H_gate(gate_error)
            self.physical_list[0].H_gate(gate_error)      

        return None

    def measure(self, basis, return_mode=False, get_operator=False, measurement_error=None):

        # Measurement
        physical_measurement_result = []
        operators = []
        for qubit in self.physical_list:
            if basis == 'Z':
                res = qubit.measureZ(get_operator=get_operator, measurement_error=measurement_error)
                if get_operator:
                    measure_result, operator = res
                    operators.append(operator)
                else:
                    measure_result = res
                physical_measurement_result.append(measure_result)
            elif basis == 'X':
                res = qubit.measureX(get_operator=get_operator, measurement_error=measurement_error)
                if get_operator:
                    measure_result, operator = res
                    operators.append(operator)
                else:
                    measure_result = res
                physical_measurement_result.append(measure_result)
            elif basis == 'Y':
                res = qubit.measureY(get_operator=get_operator, measurement_error=measurement_error)
                if get_operator:
                    measure_result, operator = res
                    operators.append(operator)
                else:
                    measure_result = res
                physical_measurement_result.append(measure_result)
            else:
                raise ValueError(f'Measurement basis: {basis} is not implemented')
        
        if get_operator:
            return bool(physical_measurement_result.count(True) % 2), operators

        # If odd True return True, even True return False
        return bool(physical_measurement_result.count(True) % 2)
    
    def measureX(self, return_mode=False, get_operator=False, measurement_error=None):
        return self.measure(basis='X', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def measureY(self, return_mode=False, get_operator=False, measurement_error=None):
        return self.measure(basis='Y', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def measureZ(self, return_mode=False, get_operator=False, measurement_error=None):
        return self.measure(basis='Z', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def error_detection_correction(self, protocol='standard', correction=True, return_syndrome=False, perfect_correction=False):

        if perfect_correction:

            error_x = []
            for i in self.physical_list:
                error_x.append(i.error_x)
            
            _ , location_x = self.classical_error_correction(error_x)
            self.physical_list[location_x].error_x = not self.physical_list[location_x].error_x

            error_z = []
            for i in self.physical_list:
                error_z.append(i.error_z)
            
            _ , location_z = self.classical_error_correction(error_z)
            self.physical_list[location_z].error_z = not self.physical_list[location_z].error_z

            self.reinitializeAncilla()
            return None

        # Generator 1
        self.ancilla_list[2].H_gate()
        self.physical_list[3].CNOT_gate(self.ancilla_list[2])
        self.physical_list[4].CNOT_gate(self.ancilla_list[2])
        self.physical_list[5].CNOT_gate(self.ancilla_list[2])
        self.physical_list[6].CNOT_gate(self.ancilla_list[2])
        self.ancilla_list[2].H_gate()
        
        # Generator 2
        self.ancilla_list[0].H_gate()
        self.physical_list[0].CNOT_gate(self.ancilla_list[0])
        self.physical_list[2].CNOT_gate(self.ancilla_list[0])
        self.physical_list[4].CNOT_gate(self.ancilla_list[0])
        self.physical_list[6].CNOT_gate(self.ancilla_list[0])
        self.ancilla_list[0].H_gate()

        # Generator 3
        self.ancilla_list[1].H_gate()
        self.physical_list[1].CNOT_gate(self.ancilla_list[1])
        self.physical_list[2].CNOT_gate(self.ancilla_list[1])
        self.physical_list[5].CNOT_gate(self.ancilla_list[1])
        self.physical_list[6].CNOT_gate(self.ancilla_list[1])
        self.ancilla_list[1].H_gate()

        # Generator 4
        self.ancilla_list[5].H_gate()
        self.physical_list[3].H_gate()
        self.physical_list[3].CNOT_gate(self.ancilla_list[5])
        self.physical_list[3].H_gate()
        self.physical_list[4].H_gate()
        self.physical_list[4].CNOT_gate(self.ancilla_list[5])
        self.physical_list[4].H_gate()
        self.physical_list[5].H_gate()
        self.physical_list[5].CNOT_gate(self.ancilla_list[5])
        self.physical_list[5].H_gate()
        self.physical_list[6].H_gate()
        self.physical_list[6].CNOT_gate(self.ancilla_list[5])
        self.physical_list[6].H_gate()
        self.ancilla_list[5].H_gate()

        # Generator 5
        self.ancilla_list[3].H_gate()
        self.physical_list[0].H_gate()
        self.physical_list[0].CNOT_gate(self.ancilla_list[3])
        self.physical_list[0].H_gate()
        self.physical_list[2].H_gate()
        self.physical_list[2].CNOT_gate(self.ancilla_list[3])
        self.physical_list[2].H_gate()
        self.physical_list[4].H_gate()
        self.physical_list[4].CNOT_gate(self.ancilla_list[3])
        self.physical_list[4].H_gate()
        self.physical_list[6].H_gate()
        self.physical_list[6].CNOT_gate(self.ancilla_list[3])
        self.physical_list[6].H_gate()
        self.ancilla_list[3].H_gate()

        # Generator 6
        self.ancilla_list[4].H_gate()
        self.physical_list[1].H_gate()
        self.physical_list[1].CNOT_gate(self.ancilla_list[4])
        self.physical_list[1].H_gate()
        self.physical_list[2].H_gate()
        self.physical_list[2].CNOT_gate(self.ancilla_list[4])
        self.physical_list[2].H_gate()
        self.physical_list[5].H_gate()
        self.physical_list[5].CNOT_gate(self.ancilla_list[4])
        self.physical_list[5].H_gate()
        self.physical_list[6].H_gate()
        self.physical_list[6].CNOT_gate(self.ancilla_list[4])
        self.physical_list[6].H_gate()
        self.ancilla_list[4].H_gate()

        syndrome_x = ''
        syndrome_z = ''
        
        for index, anc_qubit in enumerate(self.ancilla_list):
            if index < 3:
                syndrome_z += str(int(anc_qubit.measureZ()))
            else:
                syndrome_x += str(int(anc_qubit.measureZ()))

        syndrome_x = int(syndrome_x[::-1], base=2) - 1
        syndrome_z = int(syndrome_z[::-1], base=2) - 1

        if correction:
            if syndrome_x != -1:
                self.physical_list[syndrome_x].X_gate()
            
            if syndrome_z != -1:
                self.physical_list[syndrome_z].Z_gate()

            self.reinitializeAncilla()
        
        if return_syndrome:
            return syndrome_x, syndrome_z
        else:
            return None

    def reinitializeAncilla(self):

        for qubit in self.ancilla_list:
            qubit.setFree() # <--- May change in the future
            qubit.initiateTime = self.env.now

        return None

    def boolToInt(self, arr):
        r = 0
        for i in range(3):
            t = arr[i]
            r |= t << (3 - i - 1)
        return r


    def measureForFidelity(self, method='get_operator'):

        # Check look up table
        if method=='get_operator':
            operators = []
            for qubit in self.physical_list:
                operators.append(qubit.measureForFidelity())
        else:
            # Error-detection

            G = [[False, False, False, True, True, True, True], # G1
                 [False, True, True, False, False, True, True], # G2
                 [True, False, True, False, True, False, True]] # G3

            location_bool = [False, False, False]
            for i in range(len(G)):
                flag = False
                for j, qubit in enumerate(self.physical_list):
                    if qubit.error_x == G[i][j] and (qubit.error_x != False and G[i][j] != False):
                        flag = not flag
                location_bool[i] = flag
            
            location = self.boolToInt(location_bool) - 1 # minus one becasue the shift of index

            # Get logical operator
            operators = ''
            if location != -1:
                count = 0
                for qubit in self.physical_list:
                    if qubit.error_x == True:
                        count += 1
                if count % 2 == 0:
                    operators += 'X'
                else:
                    operators += 'I'
            else:
                count = 0
                for qubit in self.physical_list:
                    if qubit.error_x == True:
                        count += 1
                if count % 2 == 1:
                    operators += 'X'
                else:
                    operators += 'I'

            for i in range(len(G)):
                flag = False
                for j, qubit in enumerate(self.physical_list):
                    if qubit.error_z == G[i][j] and (qubit.error_z != False and G[i][j] != False):
                        flag = not flag
                location_bool[i] = flag
            
            location = self.boolToInt(location_bool) - 1 # minus one becasue the shift of index

            # Error correction
            # Get logical operator
            if location != -1:
                # If there are any errors, for even weight error operator,
                # after correction, return non-trivial logical operator
                count = 0
                for qubit in self.physical_list:
                    # qubit.error_z = not qubit.error_z if qubit.measurementError > random.random() else qubit.error_z
                    if qubit.error_z == True:
                        count += 1
                if count % 2 == 0:
                    operators += 'Z'
                else:
                    operators += 'I'
            else:
                # If there is no error, for odd weight error operator,
                # return, return non-trivial logical operator
                count = 0
                for qubit in self.physical_list:
                    # qubit.error_z = not qubit.error_z if qubit.measurementError > random.random() else qubit.error_z
                    if qubit.error_z == True:
                        count += 1
                if count % 2 == 1:
                    operators += 'Z'
                else:
                    operators += 'I'

        return operators

    def classical_error_correction(self, codeword):

        G = [[False, False, False, True, True, True, True], # G1
             [False, True, True, False, False, True, True], # G2
             [True, False, True, False, True, False, True]] # G3

        location_bool = [False, False, False]
        for i in range(len(G)):
            flag = False
            for j, res in enumerate(codeword):
                if res == G[i][j] and (res != False and G[i][j] != False):
                    flag = not flag
        
            location_bool[i] = flag
        
        location = self.boolToInt(location_bool) - 1 # minus one becasue the shift of index
        new_codeword = codeword[:]
        corrected = False
        if location != -1:
            corrected = True
            new_codeword[location] = not new_codeword[location]

        return new_codeword, location

    def measure(self, basis, return_mode=False, get_operator=False, measurement_error=None):

        # Measurement
        physical_measurement_result = []
        operators = []
        for qubit in self.physical_list:
            if basis == 'Z':
                res = qubit.measureZ(get_operator=get_operator, measurement_error=measurement_error)
                if get_operator:
                    measure_result, operator = res
                    operators.append(operator)
                else:
                    measure_result = res
                physical_measurement_result.append(measure_result)
            elif basis == 'X':
                res = qubit.measureX(get_operator=get_operator, measurement_error=measurement_error)
                if get_operator:
                    measure_result, operator = res
                    operators.append(operator)
                else:
                    measure_result = res
                physical_measurement_result.append(measure_result)
            elif basis == 'Y':
                res = qubit.measureY(get_operator=get_operator, measurement_error=measurement_error)
                if get_operator:
                    measure_result, operator = res
                    operators.append(operator)
                else:
                    measure_result = res
                physical_measurement_result.append(measure_result)
            else:
                raise ValueError(f'Measurement basis: {basis} is not implemented')

        # Classical error correction
        corrected_result, location = self.classical_error_correction(physical_measurement_result)

        if return_mode:
            return corrected_result, location
        
        if get_operator:
            return bool(corrected_result.count(True) % 2), operators

        # If odd True return True, even True return False
        return bool(corrected_result.count(True) % 2)
    
    def measureX(self, return_mode=False, get_operator=False, measurement_error=None):
        return self.measure(basis='X', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def measureY(self, return_mode=False, get_operator=False, measurement_error=None):
        return self.measure(basis='Y', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def measureZ(self, return_mode=False, get_operator=False, measurement_error=None):
        return self.measure(basis='Z', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def error_detection_correction(self, protocol='standard', correction=True, return_syndrome=False, perfect_correction=False):

        if perfect_correction:

            error_x = []
            for i in self.physical_list:
                error_x.append(i.error_x)
            
            _ , location_x = self.classical_error_correction(error_x)
            self.physical_list[location_x].error_x = not self.physical_list[location_x].error_x

            error_z = []
            for i in self.physical_list:
                error_z.append(i.error_z)
            
            _ , location_z = self.classical_error_correction(error_z)
            self.physical_list[location_z].error_z = not self.physical_list[location_z].error_z

            self.reinitializeAncilla()
            return None

        # Reverse-order : 2, 0, 1 ; 5, 3, 4

        # Generator 1
        self.ancilla_list[2].H_gate()
        self.physical_list[3].CNOT_gate(self.ancilla_list[2])
        self.physical_list[4].CNOT_gate(self.ancilla_list[2])
        self.physical_list[5].CNOT_gate(self.ancilla_list[2])
        self.physical_list[6].CNOT_gate(self.ancilla_list[2])
        self.ancilla_list[2].H_gate()
        
        # Generator 2
        self.ancilla_list[0].H_gate()
        self.physical_list[0].CNOT_gate(self.ancilla_list[0])
        self.physical_list[2].CNOT_gate(self.ancilla_list[0])
        self.physical_list[4].CNOT_gate(self.ancilla_list[0])
        self.physical_list[6].CNOT_gate(self.ancilla_list[0])
        self.ancilla_list[0].H_gate()

        # Generator 3
        self.ancilla_list[1].H_gate()
        self.physical_list[1].CNOT_gate(self.ancilla_list[1])
        self.physical_list[2].CNOT_gate(self.ancilla_list[1])
        self.physical_list[5].CNOT_gate(self.ancilla_list[1])
        self.physical_list[6].CNOT_gate(self.ancilla_list[1])
        self.ancilla_list[1].H_gate()

        # Generator 4
        self.ancilla_list[5].H_gate()
        self.physical_list[3].H_gate()
        self.physical_list[3].CNOT_gate(self.ancilla_list[5])
        self.physical_list[3].H_gate()
        self.physical_list[4].H_gate()
        self.physical_list[4].CNOT_gate(self.ancilla_list[5])
        self.physical_list[4].H_gate()
        self.physical_list[5].H_gate()
        self.physical_list[5].CNOT_gate(self.ancilla_list[5])
        self.physical_list[5].H_gate()
        self.physical_list[6].H_gate()
        self.physical_list[6].CNOT_gate(self.ancilla_list[5])
        self.physical_list[6].H_gate()
        self.ancilla_list[5].H_gate()

        # Generator 5
        self.ancilla_list[3].H_gate()
        self.physical_list[0].H_gate()
        self.physical_list[0].CNOT_gate(self.ancilla_list[3])
        self.physical_list[0].H_gate()
        self.physical_list[2].H_gate()
        self.physical_list[2].CNOT_gate(self.ancilla_list[3])
        self.physical_list[2].H_gate()
        self.physical_list[4].H_gate()
        self.physical_list[4].CNOT_gate(self.ancilla_list[3])
        self.physical_list[4].H_gate()
        self.physical_list[6].H_gate()
        self.physical_list[6].CNOT_gate(self.ancilla_list[3])
        self.physical_list[6].H_gate()
        self.ancilla_list[3].H_gate()

        # Generator 6
        self.ancilla_list[4].H_gate()
        self.physical_list[1].H_gate()
        self.physical_list[1].CNOT_gate(self.ancilla_list[4])
        self.physical_list[1].H_gate()
        self.physical_list[2].H_gate()
        self.physical_list[2].CNOT_gate(self.ancilla_list[4])
        self.physical_list[2].H_gate()
        self.physical_list[5].H_gate()
        self.physical_list[5].CNOT_gate(self.ancilla_list[4])
        self.physical_list[5].H_gate()
        self.physical_list[6].H_gate()
        self.physical_list[6].CNOT_gate(self.ancilla_list[4])
        self.physical_list[6].H_gate()
        self.ancilla_list[4].H_gate()

        syndrome_x = ''
        syndrome_z = ''
        
        for index, anc_qubit in enumerate(self.ancilla_list):
            if index < 3:
                syndrome_z += str(int(anc_qubit.measureZ()))
            else:
                syndrome_x += str(int(anc_qubit.measureZ()))

        syndrome_x = int(syndrome_x[::-1], base=2) - 1
        syndrome_z = int(syndrome_z[::-1], base=2) - 1

        if correction:
            if syndrome_x != -1:
                self.physical_list[syndrome_x].X_gate()
            
            if syndrome_z != -1:
                self.physical_list[syndrome_z].Z_gate()

            self.reinitializeAncilla()
        
        if return_syndrome:
            return syndrome_x, syndrome_z
        else:
            return None

    def reinitializeAncilla(self):

        for qubit in self.ancilla_list:
            qubit.setFree() # <--- May change in the future
            qubit.initiateTime = self.env.now

        return None

    def boolToInt(self, arr):
        r = 0
        for i in range(3):
            t = arr[i]
            r |= t << (3 - i - 1)
        return r


class PhysicalQubit:

    def __init__(self, node1, qubitID, qnic, role, env, table, memFunc, gate_error, measurementError, mem_model=False):
        # Simulation setup
        self.env = env
        self.table = table
        self.memoryFunc = memFunc if callable(memFunc) else lambda t: memFunc
        self.mem_model = mem_model
        self.gate_error = gate_error # Do and Dont
        self.measurementError = measurementError
        self.photonPair = None

        # network setup
        self.qubitID = f'{node1}-{qubitID}'
        self.qnics_address = qnic
        self.role = role
        self.qubit_node_address = node1
        
        self.initiateTime = None

        self.setFree()

    def setFree(self):
        self.isBusy = False
        self.partner = None
        self.partnerID = None
        self.message = None
        self.photonPair = None

        # Noise flag
        self.error_x = False
        self.error_z = False
        self.error_y = False
        self.initiateTime = None
        self.memoryProbVector = [1, 0, 0, 0]
        self.memoryErrorRate = [1, 0, 0, 0]

        self.isPurified = False
        self.last_gate_time = None

    def emitPhoton(self):
        self.photonPair = PhotonPair(self.qubitID, self.qubit_node_address, self.qnics_address)
        return self.photonPair

    def setInitialTime(self):
        self.initiateTime = self.env.now
        self.last_gate_time = self.initiateTime 
        return None

    def addXerror(self):
        self.error_x = not self.error_x
        return None

    def addZerror(self):
        self.error_z = not self.error_z
        return None

    def applySingleQubitGateError(self, prob=None):
        if prob is None:
            prob = self.memoryProbVector
        error_choice = random.choices(['I', 'X', 'Z', 'Y'], weights=prob)[0]
        if error_choice == 'I':
            pass
        elif error_choice == 'X':
            self.addXerror()
        elif error_choice == 'Z':
            self.addZerror()
        else:
            self.addXerror()
            self.addZerror()
        
        return None
    
    def applyTwoQubitGateError(self, control_qubit):

        self.applySingleQubitGateError(prob=self.gate_error)
        control_qubit.applySingleQubitGateError(prob=self.gate_error)

        return None

    def measureX(self, get_operator=False, measurement_error=None):
        
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now

        if get_operator:
            result = 0
            if self.error_x and self.error_z: # X Z
                result = 1 
            elif self.error_x and not self.error_z: # X
                result = 2
            elif self.error_z and not self.error_x: # Z
                result = 3
            # Apply measurement error
            meas_error = self.measurementError if measurement_error is None else measurement_error
            self.error_z = not self.error_z if meas_error > random.random() else self.error_z
            return self.error_z, result
        else:
            # Apply measurement error
            meas_error = self.measurementError if measurement_error is None else measurement_error
            self.error_z = not self.error_z if meas_error > random.random() else self.error_z
            return self.error_z # not

    def measureZ(self, get_operator=False, measurement_error=None):
        
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now

        if get_operator:
            result = 0
            if self.error_x and self.error_z: # X Z
                result = 1 
            elif self.error_x and not self.error_z: # X
                result = 2
            elif self.error_z and not self.error_x: # Z
                result = 3

            # Apply measurement error
            meas_error = self.measurementError if measurement_error is None else measurement_error
            self.error_x = not self.error_x if meas_error > random.random() else self.error_x
            return self.error_x, result
        else:
            # Apply measurement error
            meas_error = self.measurementError if measurement_error is None else measurement_error
            self.error_x = not self.error_x if meas_error > random.random() else self.error_x
            return self.error_x # not

    def measureY(self, get_operator=False, measurement_error=None):

        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now

        error = True
        if self.error_x and self.error_z:
            error = False
        if not self.error_x and not self.error_z:
            error = False

        if get_operator:
            result = 0
            if self.error_x and self.error_z: # X Z
                result = 1 
            elif self.error_x and not self.error_z: # X
                result = 2
            elif self.error_z and not self.error_x: # Z
                result = 3
            # Apply measurement error
            meas_error = self.measurementError if measurement_error is None else measurement_error
            error = not error if meas_error > random.random() else error
            return error, result
        else:
            # Apply measurement error
            meas_error = self.measurementError if measurement_error is None else measurement_error
            error = not error if meas_error > random.random() else error
            return error

    def measure(self, basis, get_operator=False, measurement_error=None):

        if basis in ['Z', 'z']:
            results = self.measureZ(get_operator=get_operator, measurement_error=measurement_error)
        elif basis in ['Y', 'y']:
            results = self.measureY(get_operator=get_operator, measurement_error=measurement_error)
        elif basis in ['X', 'x']:
            results = self.measureX(get_operator=get_operator, measurement_error=measurement_error)
        elif basis in ['I', 'i']:
            results = False

        return results

    def I_gate(self, gate_error=None):

        # Apply Pauli error piror to application of gate depend on time
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now

        g_e = (1 - self.gate_error[0]) if gate_error is None else gate_error
        if g_e < random.random():
            pass
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])

        return None

    def H_gate(self, gate_error=None):

        # Apply Pauli error piror to application of gate depend on time
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now
        
        g_e = (1 - self.gate_error[0]) if gate_error is None else gate_error
        if g_e < random.random():
            self.error_z, self.error_x = self.error_x, self.error_z
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
    
        return None

    def X_gate(self, gate_error=None):

        # Apply Pauli error piror to application of gate depend on time
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now
        
        g_e = (1 - self.gate_error[0]) if gate_error is None else gate_error
        if g_e < random.random():
            self.error_x = not self.error_x
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
    
        return None
    
    def Z_gate(self, gate_error=None):

        # Apply Pauli error piror to application of gate depend on time
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now
        
        g_e = (1 - self.gate_error[0]) if gate_error is None else gate_error
        if g_e < random.random():
            self.error_z = not self.error_z
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
        
        return None

    def S_dagger_gate(self, gate_error=None):

        # Apply Pauli error piror to application of gate depend on time
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now

        g_e = (1 - self.gate_error[0]) if gate_error is None else gate_error
        if g_e < random.random():
            if self.error_x:
                self.error_z = not self.error_z 
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])

        return None

    def S_gate(self, gate_error=None):

        # Apply Pauli error piror to application of gate depend on time
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now

        g_e = (1 - self.gate_error[0]) if gate_error is None else gate_error
        if g_e < random.random():
            if self.error_x:
                self.error_z= not self.error_z
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])

        return None

    def CNOT_gate(self, control_qubit, gate_error=None):

        # Apply Pauli error piror to application of gate depend on time
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            control_qubit.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now
            control_qubit.last_gate_time = control_qubit.env.now
        
        g_e = (1 - self.gate_error[0]) if gate_error is None else gate_error
        if g_e < random.random():
        
            if control_qubit.error_x:
                self.error_x = not self.error_x
            
            if self.error_z:
                control_qubit.error_z = not control_qubit.error_z

        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
            control_qubit.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
        
        return None

    def CZ_gate(self, control_qubit, gate_error):

        # Apply Pauli error piror to application of gate depend on time
        if self.mem_model:
            self.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            control_qubit.applySingleQubitGateError(prob=self.memoryFunc(self.env.now - self.last_gate_time))
            self.last_gate_time = self.env.now
            control_qubit.last_gate_time = control_qubit.env.now

        #self.H_gate(gate_error=gate_error)
        #self.CNOT_gate(control_qubit, gate_error=gate_error)
        #self.H_gate(gate_error=gate_error)

        # Direct error propagation (need to check)
        g_e = (1 - self.gate_error[0]) if gate_error is None else gate_error
        if g_e < random.random():
            if self.error_x:
                control_qubit.error_z = not control_qubit.error_z

            if control_qubit.error_x:
                self.error_z = not self.error_z
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
            control_qubit.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])

    def applyMemoryError2(self):

        # Update self.memoryProbVector according to self.memoryFunc
        self.memoryProbVector = self.memoryFunc(self.env.now - self.initiateTime)

        return None

    def measureForFidelity(self, method=None, basis=None):
        self.applyMemoryError2()
        self.applySingleQubitGateError()

        result = 0
        if self.error_x and self.error_z: # X Z
            result = 1 
        elif self.error_x and not self.error_z: # X
            result = 2
        elif self.error_z and not self.error_x: # Z
            result = 3

        return result

class PhotonPair:

    def __init__(self, qubitID, node, QNICAddress) -> None:
        self.matterID = qubitID
        self.matterNode = node
        self.matterQNICAddress = QNICAddress

 