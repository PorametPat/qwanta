from .qubit import PhysicalQubit
import random

def GetQubit(memoryFunction=None, gate_error=None, measurementError=None):
    
    node = 'DummyNode'
    qubitID = 'Qubit1'
    qnic = 'QNIC1'
    role = 'Physical'
    table = 'Table1'
    env = VirtualEnv()
    memoryFunction = memoryFunction if memoryFunction is not None else [1, 0, 0, 0]
    gate_error = gate_error if gate_error is not None else 0
    measurementError = measurementError if measurementError is not None else 0

    q = PhysicalQubit(node, qubitID, qnic, role, env, table, memoryFunction, gate_error, measurementError)
    q.setInitialTime()
    
    return q

def GetQubit_experimental(gate_error=0, memory_function=[1, 0, 0, 0], measurement_error=0):

    env = VirtualEnv()
    q = Qubit(env, gate_error, memory_function, measurement_error)
    q.setInitialTime()

    return q

class Qubit:

    def __init__(self, env, gate_error=0, memory_function=[1, 0, 0, 0], measurement_error=0) -> None:
        self.env = env 
        self.memory_function = memory_function if callable(memory_function) else lambda t: memory_function
        self.gate_error = gate_error
        self.measurement_error = measurement_error
        self.id = None

        self.setFree()

    def setFree(self):

        # Noise flag
        self.error_x = False
        self.error_z = False
        self.error_y = False
        self.initiateTime = None
        self.memoryProbVector = [1, 0, 0, 0]
        self.memoryErrorRate = [1, 0, 0, 0]

        self.isPurified = False
        self.last_gate_time = None

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

    def applySingleQubitError(self, prob):
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

    def measure(self, basis, measurement_error=None):

        # Apply memory error
        self.applySingleQubitError(self.memory_function(self.env.now - self.initiateTime))

        meas_error = self.measurementError if measurement_error is None else measurement_error
        
        if basis == 'X':
            return not self.error_z if meas_error > random.random() else self.error_z

        elif basis == 'Y':
            error = True
            if self.error_x and self.error_z:
                error = False
            if not self.error_x and not self.error_z:
                error = False
            
            return not error if meas_error > random.random() else error

        elif basis == 'Z':
            return not self.error_x if meas_error > random.random() else self.error_x
        
        else:
            raise ValueError(f'Measured basis is not implemented: {basis}')

    def SingleQubitGate(self, instruc, gate_error=None):

        g_e = self.gate_error if gate_error is None else gate_error

        if (1 - g_e) > random.random():
            if instruc in ['I', 'i']:
                pass
            elif instruc in ['H', 'h']:
                self.error_z, self.error_x = self.error_x, self.error_z
            elif instruc in ['X', 'x']:
                self.error_x = not self.error_x
            elif instruc in ['Z', 'z']:
                self.error_z = not self.error_z
            elif instruc in ['S', 's', 'Sdg', 'sdg']:
                if self.error_x:
                    self.error_z = not self.error_z 
            else:
                raise ValueError(f'Gate is not implemented: {instruc}')
            
        else:
            self.applySingleQubitError(prob=[0.25, 0.25, 0.25, 0.25])

        return None

    def TwoQubitsGate(self, target_qubit, instruc, gate_error=None):
        '''
            Call signature control_qubit.TwoQubitGate(target_qubit)
        '''
        
        g_e = self.gate_error if gate_error is None else gate_error
        if (1 - g_e) > random.random():
            if self.error_x:
                target_qubit.error_x = not target_qubit.error_x
            
            if target_qubit.error_z:
                self.error_z = not self.error_z
        else:
            self.applySingleQubitError(prob=[0.25, 0.25, 0.25, 0.25])
            target_qubit.applySingleQubitError(prob=[0.25, 0.25, 0.25, 0.25])

class VirtualEnv:
    """
    Create initialize information for physical qubit
    """
    def __init__(self, time=0):
        self.now = time

class DirectFidelityEstimator:

    def __init__(self, stabilizers) -> None:


        # Check if stabilizers contain identity or not
        # if conained exclude it

        for stab in stabilizers:
            if stab.count('I') == len(stab):
                stabilizers.remove(stab)

        self.stabilizers = stabilizers
        self.n_stabilizers = len(stabilizers) + 1

        self.expectation_values = {
            stab : {
                'commute': 0,
                'anti-commute': 0
            }
        for stab in self.stabilizers}

        self.num_measured = 0

    def add_readout(self, readout, stabilizer):

        # Check if readout commute with stabilizer 
        is_that_commute = self.is_commute(readout, stabilizer)
        if is_that_commute:
            self.expectation_values[stabilizer]['commute'] += 1
        else:
            self.expectation_values[stabilizer]['anti-commute'] += 1

        self.num_measured += 1

    def measure(self, qubits):

        if not isinstance(qubits, list):
            qubits = [qubits]

        # Randomly choose stabilizer to measure
        stabilizer = random.choices(self.stabilizers)[0]

        readout = []
        for stab, qubit in zip(stabilizer, qubits):
            readout.append(qubit.measure(basis=stab))
                
        # Check if readout commute with stabilizer 
        is_that_commute = self.is_commute(readout, stabilizer)
        if is_that_commute:
            self.expectation_values[stabilizer]['commute'] += 1
        else:
            self.expectation_values[stabilizer]['anti-commute'] += 1

        self.num_measured += 1

        return True

    def is_commute(self, readout, stabilizer):

        result = 0
        for ro, stab in zip(readout, stabilizer):
            if (stab != 'I') and ro:
                # Anti-commute
                result += 1

        return False if result % 2 == 1 else True

    def estimate_fidelity(self):

        # Check if fidelity could be estimate
        summation = 0
        for stab in self.expectation_values:

            if (self.expectation_values[stab]['commute'] + self.expectation_values[stab]['anti-commute']) < 1:
                return None

            exp_val = (self.expectation_values[stab]['commute'] - \
                       self.expectation_values[stab]['anti-commute']) \
                       /(self.expectation_values[stab]['commute'] + \
                       self.expectation_values[stab]['anti-commute'])
            
            summation += exp_val

        self.fidelity = (1 / self.n_stabilizers)*(1 + summation)

        return self.fidelity