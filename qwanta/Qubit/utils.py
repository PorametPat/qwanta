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