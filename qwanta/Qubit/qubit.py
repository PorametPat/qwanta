from logging import error
import random
import numpy as np 
from typing import List, Union, Any, Callable, Optional

class LogicalQubit:

    def __init__(self, 
                 node1: Any, 
                 qubitID: Any, 
                 qnic: Any, 
                 env: Any):
        """
        Init of Logical qubit

        Args:
            node1 (Any): Node that this logical qubit located
            qubitID (Any): Part of ID of this logical qubit to be combine with node 1
            qnic (Any): QNIC that this logical qubit located
            env (Any): Environment of this qubit.
        """
        # Simulation setup
        self.env = env

        # Network setup
        self.qubitID = f'{node1}-{qubitID}'
        self.qnics_address = qnic
        self.qubit_node_address = node1

        # Physical List
        self.physical_list = []
        self.ancilla_list = []
        self.entangle_partner = []

    def setFree(self):
        
        for qubit in self.physical_list:
            qubit.setFree()

        return None

    def I_gate(self, gate_error: Optional[float] = None, ecc: Optional[str] = 'Steane'):
        """
        Apply logical Identity gate to this logical qubit

        Args:
            gate_error (Optional[float], optional): Probablity of gate error which will override physical qubit default. Defaults to None.
            ecc (Optional[str], optional): Type of error correction to be used. Defaults to 'Steane'.

        Returns:
            None:
        """
        
        if ecc == 'Steane':
            for i in self.physical_list:
                i.I_gate(gate_error=gate_error)
        elif ecc == 'Prototype':
            for i in self.physical_list:
                i.Prototype_I_gate(gate_error=gate_error)
        return None

    def H_gate(self, gate_error: Optional[float] = None, ecc: Optional[str] = 'Steane'):
        """
        Apply logical Hadamard gate to this logical qubit

        Args:
            gate_error (Optional[float], optional): Probablity of gate error which will override physical qubit default. Defaults to None.
            ecc (Optional[str], optional): Type of error correction to be used. Defaults to 'Steane'.

        Returns:
            None:
        """

        if ecc == 'Steane':
            for i in self.physical_list:
                i.H_gate(gate_error=gate_error)
        elif ecc == 'Prototype':
            for i in self.physical_list:
                i.Prototype_H_gate(gate_error=gate_error)
        return None

    def X_gate(self, gate_error: Optional[float] = None, ecc: Optional[str] = 'Steane'):
        """
        Apply logical Pauli X gate to this logical qubit

        Args:
            gate_error (Optional[float], optional): Probablity of gate error which will override physical qubit default. Defaults to None.
            ecc (Optional[str], optional): Type of error correction to be used. Defaults to 'Steane'.

        Returns:
            None:
        """

        if ecc == 'Steane':
            for i in self.physical_list:
                i.X_gate(gate_error=gate_error)
        elif ecc == 'Prototype':
            for i in self.physical_list:
                i.Prototype_X_gate(gate_error=gate_error)
        return None

    def Z_gate(self, gate_error: Optional[float] = None, ecc: Optional[str] = 'Steane'):
        """
        Apply logical Pauli Z gate to this logical qubit

        Args:
            gate_error (Optional[float], optional): Probablity of gate error which will override physical qubit default. Defaults to None.
            ecc (Optional[str], optional): Type of error correction to be used. Defaults to 'Steane'.

        Returns:
            None:
        """

        if ecc == 'Steane':
            for i in self.physical_list:
                i.Z_gate(gate_error=gate_error)
        elif ecc == 'Prototype':
            for i in self.physical_list:
                i.Prototype_Z_gate(gate_error=gate_error)
        return None

    def S_dagger_gate(self, gate_error: Optional[float] = None, ecc: Optional[str] = 'Steane'):
        """
        Apply logical S dagger gate to this logical qubit

        Args:
            gate_error (Optional[float], optional): Probablity of gate error which will override physical qubit default. Defaults to None.
            ecc (Optional[str], optional): Type of error correction to be used. Defaults to 'Steane'.

        Returns:
            None:
        """

        if ecc == 'Steane':
            for i in self.physical_list:
                i.S_dagger_gate(gate_error=gate_error)
        elif ecc == 'Prototype':
            for i in self.physical_list:
                i.Prototype_S_dagger_gate(gate_error=gate_error)
        return None

    def S_gate(self, gate_error: Optional[float] = None, ecc: Optional[str] = 'Steane'):
        """
        Apply logical S gate to this logical qubit

        Args:
            gate_error (Optional[float], optional): Probablity of gate error which will override physical qubit default. Defaults to None.
            ecc (Optional[str], optional): Type of error correction to be used. Defaults to 'Steane'.

        Returns:
            None:
        """

        if ecc == 'Steane':
            for i in self.physical_list:
                i.S_gate(gate_error=gate_error)
        elif ecc == 'Prototype':
            for i in self.physical_list:
                i.Prototype_S_gate(gate_error=gate_error)
        return None

    def CNOT_gate(self, 
                  control_qubits: List, 
                  target_qubits: List, 
                  ecc: Optional[str] = 'Steane', 
                  gate_error: Optional[float] = None):
        """
        Apply logical CNOT gate to control and target qubits

        Args:
            control_qubits (List): List of physical qubits of control logical qubit
            target_qubits (List): List of physical qubits of target logical qubit
            ecc (Optional[str], optional): Probablity of gate error which will override physical qubit default. Defaults to None.
            gate_error (Optional[float], optional): Type of error correction to be used. Defaults to 'Steane'.

        Returns:
            None:
        """

        if ecc == 'Steane':
            for i in range(len(control_qubits)):
                target_qubits[i].CNOT_gate(control_qubits[i], gate_error=gate_error)
        elif ecc == 'Prototype':
            for i in range(len(control_qubits)):
                target_qubits[i].Prototype_CNOT_gate(control_qubits[i], gate_error=gate_error)
        return None

    def encode(self, ecc: Optional[str] = 'Steane'):
        """
        Encode logical qubit using self.physical_list
        For Steane code, input qubit is 0 but will be swap to qubit 2 internally.

        Args:
            ecc (Optional[str], optional): Type of quantum error correction code to used. Defaults to 'Steane'.

        Returns:
            None:
        """

        if ecc == 'Steane':
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
        
        elif ecc == 'Prototype':
            # Swap input to index 2
            self.physical_list[0], self.physical_list[2] = self.physical_list[2], self.physical_list[0]

            # Hard code
            self.physical_list[0].Prototype_H_gate()
            self.physical_list[1].Prototype_H_gate()
            self.physical_list[3].Prototype_H_gate()

            self.physical_list[4].Prototype_CNOT_gate(self.physical_list[2])
            self.physical_list[5].Prototype_CNOT_gate(self.physical_list[2])

            self.physical_list[2].Prototype_CNOT_gate(self.physical_list[0])
            self.physical_list[4].Prototype_CNOT_gate(self.physical_list[0])
            self.physical_list[6].Prototype_CNOT_gate(self.physical_list[0])

            self.physical_list[2].Prototype_CNOT_gate(self.physical_list[1])
            self.physical_list[5].Prototype_CNOT_gate(self.physical_list[1])
            self.physical_list[6].Prototype_CNOT_gate(self.physical_list[1])

            self.physical_list[4].Prototype_CNOT_gate(self.physical_list[3])
            self.physical_list[5].Prototype_CNOT_gate(self.physical_list[3])
            self.physical_list[6].Prototype_CNOT_gate(self.physical_list[3])

        return None


    def decode(self, ecc='Steane', no_error=False):

        gate_error = 0 if no_error else None

        if ecc == 'Steane':
           
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

    def classical_error_correction(self, codeword: List):
        """
        Method for classical error correction on input codeword using Steane generator.

        Args:
            codeword (List): List of measurement result of each qubit in self.physical_list

        Returns:
            (List, int): return a corrected codeword and position of error detected, -1 if there is no error.
        """

        # Check commutation with Generators
        G = [[True, False, True, False, True, False, True], # G2
             [False, True, True, False, False, True, True], # G3
             [False, False, False, True, True, True, True]] # G1
        location_bool = [False, False, False]
        for i in range(len(G)):
            flag = False
            for j, res in enumerate(codeword):
                if res == G[i][j] and (res != False and G[i][j] != False):
                    flag = not flag
        
            location_bool[i] = flag
        
        # Convert commutation result into location of noise, if -1 then there is no error.
        location = int(''.join([str(int(i)) for i in location_bool])[::-1], base=2) - 1  # minus one becasue the shift of index
        
        # Correct codeword if there is an error.
        new_codeword = codeword[:]
        corrected = False
        if location != -1:
            corrected = True
            new_codeword[location] = not new_codeword[location]

        return new_codeword, location

    def measure(self, 
                basis: str, 
                return_mode: Optional[bool] = False, 
                get_operator: Optional[bool] = False, 
                measurement_error: Optional[float] = None):
        """
        Measure logical qubit 

        Args:
            basis (str): measured-basis
            return_mode (Optional[bool], optional): If True, will return both corrected codeword and location of error. Defaults to False.
            get_operator (Optional[bool], optional): If True, will return decoded result and raw result. Defaults to False.
            measurement_error (Optional[float], optional): Probablity of measurement error to be override physical qubit. Defaults to None.

        Raises:
            ValueError: Invalid measurement-basis

        Returns:
            bool: Return classical error corrected decoded result of this logical qubit
        """

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
    
    def measureX(self, 
                 return_mode: Optional[bool] = False, 
                 get_operator: Optional[bool] = False, 
                 measurement_error: Optional[float] = None):
        """Interface of logical qubit measurement in X basis

        Args:
            return_mode (Optional[bool], optional): If True, will return both corrected codeword and location of error. Defaults to False.
            get_operator (Optional[bool], optional): If True, will return decoded result and raw result. Defaults to False.
            measurement_error (Optional[float], optional): Probablity of measurement error to be override physical qubit. Defaults to None.

        Returns:
            bool: Return classical error corrected decoded result of measuring this logical qubit in X basis
        """
        return self.measure(basis='X', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def measureY(self, 
                 return_mode: Optional[bool] = False, 
                 get_operator: Optional[bool] = False, 
                 measurement_error: Optional[float] = None):
        """Interface of logical qubit measurement in Y basis

        Args:
            return_mode (Optional[bool], optional): If True, will return both corrected codeword and location of error. Defaults to False.
            get_operator (Optional[bool], optional): If True, will return decoded result and raw result. Defaults to False.
            measurement_error (Optional[float], optional): Probablity of measurement error to be override physical qubit. Defaults to None.

        Returns:
            bool: Return classical error corrected decoded result of measuring this logical qubit in Y basis
        """       
        return self.measure(basis='Y', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def measureZ(self, 
                 return_mode: Optional[bool] = False, 
                 get_operator: Optional[bool] = False, 
                 measurement_error: Optional[float] = None):
        """Interface of logical qubit measurement in Z basis

        Args:
            return_mode (Optional[bool], optional): If True, will return both corrected codeword and location of error. Defaults to False.
            get_operator (Optional[bool], optional): If True, will return decoded result and raw result. Defaults to False.
            measurement_error (Optional[float], optional): Probablity of measurement error to be override physical qubit. Defaults to None.

        Returns:
            bool: Return classical error corrected decoded result of measuring this logical qubit in Z basis
        """    
        return self.measure(basis='Z', return_mode=return_mode, get_operator=get_operator, measurement_error=measurement_error)

    def error_detection_correction(self, 
                                   protocol: Optional[str] = 'standard', 
                                   correction: Optional[bool] = True, 
                                   return_syndrome: Optional[bool] = False, 
                                   perfect_correction: Optional[bool] = False):
        """Quantum error correction to this logical qubit and reinitialize ancilla qubits too.

        Args:
            protocol (Optional[str], optional): Protocol for QEC. Defaults to 'standard'.
            correction (Optional[bool], optional): If False, not apply X or Z gate according to error detected 
                                                   and not reinitialize ancilla qubits. Defaults to True.
            return_syndrome (Optional[bool], optional): If True, return syndrome of syndrome measurement both X and Z basis. 
                                                        Defaults to False.
            perfect_correction (Optional[bool], optional): If True, perform perfect quantum error correction directly 
                                                           using classical error correction method. Defaults to False.

        Returns:
            None: 
        """

        if perfect_correction:

            error_x = []
            for i in self.physical_list:
                error_x.append(i.error_x)
            
            _ , location_x = self.classical_error_correction(error_x)
            if location_x != -1:
                self.physical_list[location_x].error_x = not self.physical_list[location_x].error_x

            error_z = []
            for i in self.physical_list:
                error_z.append(i.error_z)
            
            _ , location_z = self.classical_error_correction(error_z)
            if location_z != -1:
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
        """Reinitialize ancilla qubits

        Returns:
            None: 
        """

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
    """
    Module of error tracking based physical qubit 
    """
    def __init__(self, 
                 node1: Any, 
                 node2: Any, 
                 qubitID: Any, 
                 qnic: Any, 
                 role: Any, 
                 env: Any, 
                 table: Any, 
                 memFunc: Union[Callable, List], 
                 gate_error: Union[int, float], 
                 measurementError: Union[int, float]):
        """The init function of physical qubit.

        Args:
            node1 (Any): A node that this qubit is located.
            node2 (Any): A neighbor node that this qubit is located.
            qubitID (Any): A part of qubit ID that will be combine with node1.
            qnic (Any): A quantum network interface caed that this qubit located.
            role (Any): A role of qubit whether it is external or internal qubit.
            env (Any): An environment of qubit, should have env.now for calculate probablity of memory error.
            table (Any): A table that qubit is located.
            memFunc (Union[Callable, List]): A memory function for qubit.
            gate_error (Union[int, float]): A propability of gate error of qubit
            measurementError (Union[int, float]): A propability of measurement error of qubit
        """

        # Simulation setup
        self.env = env
        self.table = table
        self.memoryFunc = memFunc if callable(memFunc) else lambda t: memFunc
        self.gate_error = gate_error # Do and Dont
        self.measurementError = measurementError
        self.photonPair = None

        # network setup
        self.qubitID = f'{node1}-{qubitID}'
        self.qnics_address = qnic
        self.role = role
        self.qubit_node_address = node1
        self.qubit_neighbor_address = node2
        
        self.initiateTime = None

        self.setFree()

    def setFree(self):
        """
        Method that reset status of the qubit
        """
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
        """
        Method for initial time for qubit

        Returns:
            None: 
        """
        self.initiateTime = self.env.now
        self.last_gate_time = self.initiateTime 
        return None


    def addXerror(self):
        """
        Add Pauli-X error to the qubit

        Returns:
            None: 
        """
        self.error_x = not self.error_x
        return None

    def addZerror(self):
        """
        Add Pauli-Z error to the qubit

        Returns:
            None: 
        """
        self.error_z = not self.error_z
        return None

    def applySingleQubitGateError(self, prob: Optional[List] = None):
        """
        Apply single-qubit gate error to the qubit with some probablity

        Args:
            prob (Optional[List], optional): If prob is provied, it will 
                                             be used as distribution for apply Pauli error. 
                                             Defaults to None.

        Returns:
            None: 
        """
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

    def measureX(self, get_operator: Optional[bool] = False, measurement_error: Optional[float] = None):
        """
        Apply measurement error to qubit then, measure qubit in X-basis.

        Args:
            get_operator (Optional[bool], optional): If True, will return also an integer indicate a Pauli error on qubit. 
                                                     Defaults to False.
            measurement_error (Optional[float], optional): measurement error to override the qubit default. Defaults to None.

        Returns:
            bool: Whether there is bit-flip error in X-basis or not.
        """

        self.applySingleQubitGateError(self.memoryFunc(self.env.now - self.initiateTime))

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

    def measureZ(self, get_operator: Optional[bool] = False, measurement_error: Optional[float] = None):
        """
        Apply measurement error to qubit then, measure qubit in Z-basis.

        Args:
            get_operator (Optional[bool], optional): If True, will return also an integer indicate a Pauli error on qubit. 
                                                     Defaults to False.
            measurement_error (Optional[float], optional): measurement error to override the qubit default. Defaults to None.

        Returns:
            bool: Whether there is bit-flip error in Z-basis or not.
        """

        self.applySingleQubitGateError(self.memoryFunc(self.env.now - self.initiateTime))

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

    def measureY(self, get_operator: Optional[bool] = False, measurement_error: Optional[float] = None):
        """
        Apply measurement error to qubit then, measure qubit in Y-basis.

        Args:
            get_operator (Optional[bool], optional): If True, will return also an integer indicate a Pauli error on qubit. 
                                                     Defaults to False.
            measurement_error (Optional[float], optional): measurement error to override the qubit default. Defaults to None.

        Returns:
            bool: Whether there is bit-flip error in Y-basis or not.
        """

        self.applySingleQubitGateError(self.memoryFunc(self.env.now - self.initiateTime))

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

    def measure(self, basis: str, get_operator: Optional[bool] = False, measurement_error: Optional[float] = None):
        """
        Interface for apply measurement error and measure qubit in particular basis.

        Args:
            basis (str): basis for qubit to be measured.
            get_operator (Optional[bool], optional): If True, will return also an integer indicate a Pauli error on qubit. 
                                                     Defaults to False.
            measurement_error (Optional[float], optional): measurement error to override the qubit default. Defaults to None.

        Returns:
            bool: Whether there is bit-flip error in measured basis or not.
        """

        if basis in ['Z', 'z']:
            results = self.measureZ(get_operator=get_operator, measurement_error=measurement_error)
        elif basis in ['Y', 'y']:
            results = self.measureY(get_operator=get_operator, measurement_error=measurement_error)
        elif basis in ['X', 'x']:
            results = self.measureX(get_operator=get_operator, measurement_error=measurement_error)
        elif basis in ['I', 'i']:
            results = False

        return results

    def I_gate(self, gate_error: Optional[float] = None, mem_model: Optional[bool] = False):
        """
        Apply Identity gate to qubit, with some probablity apply depolarizing channel instead.

        Args:
            gate_error (Optional[float], optional): Probablity of gate error to be overried qubit default. Defaults to None.
            mem_model (Optional[bool], optional): Memory model to be used, Currently not implemented. Defaults to False.

        Returns:
            None: 
        """
        # Apply Pauli error piror to application of gate depend on time
        if mem_model:
            pass

        g_e = self.gate_error if gate_error is None else gate_error
        if (1 - g_e) > random.random():
            pass
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])

        return None

    def Prototype_I_gate(self, gate_error=None):

        g_e = self.gate_error if gate_error is None else gate_error
        if random.random() < g_e:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
    
        return None

    def H_gate(self, gate_error: Optional[float] = None):
        """
        Apply Hadamard gate to qubit, with some probablity apply depolarizing channel instead.

        Args:
            gate_error (Optional[float], optional): Probablity of gate error to be overried qubit default. Defaults to None.

        Returns:
            None:
        """
        
        g_e = self.gate_error if gate_error is None else gate_error
        if (1 - g_e) > random.random():
            self.error_z, self.error_x = self.error_x, self.error_z
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
    
        return None

    def Prototype_H_gate(self, gate_error=None):

        self.error_z, self.error_x = self.error_x, self.error_z
        g_e = self.gate_error if gate_error is None else gate_error
        if random.random() < g_e:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
    
        return None

    def X_gate(self, gate_error: Optional[float] = None):
        """
        Apply Pauli-X gate to qubit, with some probablity apply depolarizing channel instead.

        Args:
            gate_error (Optional[float], optional): Probablity of gate error to be overried qubit default. Defaults to None.

        Returns:
            None:
        """
        
        g_e = self.gate_error if gate_error is None else gate_error
        if (1 - g_e) > random.random():
            self.error_x = not self.error_x
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
    
        return None

    def Prototype_X_gate(self, gate_error=None):

        self.error_x = not self.error_x
        g_e = self.gate_error if gate_error is None else gate_error
        if random.random() < g_e:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
    
        return None
    
    def Z_gate(self, gate_error: Optional[float] = None):
        """
        Apply Pauli-Z gate to qubit, with some probablity apply depolarizing channel instead.

        Args:
            gate_error (Optional[float], optional): Probablity of gate error to be overried qubit default. Defaults to None.

        Returns:
            None:
        """
        
        g_e = self.gate_error if gate_error is None else gate_error
        if (1 - g_e) > random.random():
            self.error_z = not self.error_z
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
        
        return None

    def Prototype_Z_gate(self, gate_error=None):

        self.error_z = not self.error_z
        g_e = self.gate_error if gate_error is None else gate_error
        if random.random() < g_e:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
    
        return None

    def S_dagger_gate(self, gate_error: Optional[float] = None):
        """
        Apply S gate to qubit, with some probablity apply depolarizing channel instead.

        Args:
            gate_error (Optional[float], optional): Probablity of gate error to be overried qubit default. Defaults to None.

        Returns:
            None:
        """

        g_e = self.gate_error if gate_error is None else gate_error
        if (1 - g_e) > random.random():
            if self.error_x:
                self.error_z = not self.error_z 
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])

        return None

    def Prototype_S_dagger_gate(self, gate_error=None):

        if self.error_x:
            self.error_z = not self.error_z 
        g_e = self.gate_error if gate_error is None else gate_error
        if random.random() < g_e:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])

        return None

    def S_gate(self, gate_error: Optional[float] = None):
        """
        Apply S gate to qubit, with some probablity apply depolarizing channel instead.

        Args:
            gate_error (Optional[float], optional): Probablity of gate error to be overried qubit default. Defaults to None.

        Returns:
            None:
        """

        g_e = self.gate_error if gate_error is None else gate_error
        if (1 - g_e) > random.random():
            if self.error_x:
                self.error_z= not self.error_z
        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])

        return None

    def CNOT_gate(self, control_qubit: Any,  gate_error: Optional[float] = None):
        """
        Apply CNOT gate to qubit as a target, with some probablity apply depolarizing channel instead.

        Args:
            control_qubit (Any): Control qubit for CNOT gate
            gate_error (Optional[float], optional): Probablity of gate error to be overried qubit default. Defaults to None.

        Returns:
            None:
        """
        
        g_e = self.gate_error if gate_error is None else gate_error
        if (1 - g_e) > random.random():
        
            if control_qubit.error_x:
                self.error_x = not self.error_x
            
            if self.error_z:
                control_qubit.error_z = not control_qubit.error_z

        else:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
            control_qubit.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
        
        return None

    def Prototype_CNOT_gate(self, control_qubit, gate_error=None):
        
        if control_qubit.error_x:
                self.error_x = not self.error_x   
        if self.error_z:
            control_qubit.error_z = not control_qubit.error_z

        g_e = self.gate_error if gate_error is None else gate_error
        if random.random() < g_e:
            self.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
            control_qubit.applySingleQubitGateError(prob=[0.25, 0.25, 0.25, 0.25])
        
        return None

class PhotonPair:

    def __init__(self, qubitID, node, QNICAddress) -> None:
        self.matterID = qubitID
        self.matterNode = node
        self.matterQNICAddress = QNICAddress

 