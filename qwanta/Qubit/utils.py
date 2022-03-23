from .qubit import PhysicalQubit
import random
from sympy import symbols
import pandas as pd

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

        meas_error = self.measurement_error if measurement_error is None else measurement_error
        
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
            elif instruc in ['Y', 'y']:
                self.error_x = not self.error_x
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

class ProbablyQuantumCircuit:

    def __init__(self, n_qubit):
        
        self.n_qubit = n_qubit
        self.qubits = [GetQubit() for _ in range(n_qubit)]

        self.instructions = []

        self.Pauli1 = ['I', 'X', 'Y', 'Z']
        self.Pauli2 = [ i + j for i in self.Pauli1 for j in self.Pauli1]
    
    def update_instruction(self, instruc, qubits, prob_var, variation):
        self.instructions.append({
            'instruction': instruc,
            'qubits' : qubits, 
            'prob_var': prob_var,
            'variation': variation
        })

    def sqg(self, gate, q_index, var='p_gate1'):
        var = Symbol(var) if isinstance(var, str) else var
        variation = ['None'] if var == 0 else ['None'] + self.Pauli1
        self.update_instruction(gate, q_index, var, variation)

    def tqg(self, gate, cq_index, tq_index, var='p_gate2'):
        var = Symbol(var) if isinstance(var, str) else var
        variation = ['None'] if var == 0 else ['None'] + self.Pauli2
        self.update_instruction(gate, [cq_index, tq_index], var, variation)

    def mem_error(self, q_index, var='p_mem'):
        var = Symbol(var) if isinstance(var, str) else var
        variation = ['None'] if var == 0 else ['None', 'X', 'Y', 'Z']
        self.update_instruction('memory', q_index, var, variation)
    
    def dep_error(self, q_index, var='p_dep'):
        var = Symbol(var) if isinstance(var, str) else var
        variation = ['None'] if var == 0 else ['None', 'X', 'Y', 'Z']
        self.update_instruction('depolarizing', q_index, var, variation)

    def measure(self, q_index, basis='Z', var='p_meas'):
        var = Symbol(var) if isinstance(var, str) else var
        variation = ['None'] if var == 0 else ['None', 'Flip']
        self.update_instruction('measure ' + basis, q_index, var, variation)

    def c_if(self, gate, cq_index, tq_index, var='p_gate1'):
        # model this instruction as variation
        # with probability 1/2 apply gate to q_index
        var = Symbol(var) if isinstance(var, str) else var
        variation = ['No gate', 'None'] if var == 0 else ['No gate', 'None'] + self.Pauli1
        self.update_instruction('c_if ' + gate, [cq_index, tq_index], var, ['No gate', 'None'] + self.Pauli1)

    def syndrome(self):
        pass

    def evalute(self, errors, instructions=None, stabilizer=None, index_stabilizer=None):

        instructions = self.instructions if instructions is None else instructions
        circuit = pd.DataFrame(instructions)

        prob = 1
        qubits = [GetQubit_experimental() for _ in range(self.n_qubit)]
        measurement_readout = {}
        for instruc, qubits_index, var,  error in zip(circuit['instruction'], circuit['qubits'] , circuit['prob_var'], errors):

            if instruc in ['i', 'h', 'x', 'z', 's', 'sdg']:
                if error == 'None':
                    qubits[qubits_index].SingleQubitGate(instruc=instruc)
                    prob *= (1 - var)
                else:
                    qubits[qubits_index].SingleQubitGate(instruc=error)
                    prob *= (1/4)*var
            if instruc in ['cx']:
                if error == 'None':
                    qubits[qubits_index[0]].TwoQubitsGate(qubits[qubits_index[1]], instruc)
                    prob *= (1 - var)
                else:
                    qubits[qubits_index[0]].SingleQubitGate(error[0])
                    qubits[qubits_index[1]].SingleQubitGate(error[1])
                    prob *= (1/16)*var
            if instruc in ['depolarizing']:

                if error == 'None':
                    prob *= (1 - var)
                else:
                    qubits[qubits_index].SingleQubitGate(instruc=error)
                    prob *= (1/3)*var

            if instruc in ['memory']:

                if error == 'None':
                    prob *= var
                else:
                    qubits[qubits_index].SingleQubitGate(instruc=error)
                    prob *= (1/3)*(1 - var)
            
            if instruc.split(' ')[0] == 'measure':
                basis = instruc.split(' ')[1]

                if basis != 'I':
                    result = qubits[qubits_index].measure(basis)

                    if error == 'Flip':
                        result = not result
                        prob *= var
                    else:
                        prob *= (1 - var)
                else:
                    result = False
                
                measurement_readout[qubits_index] = result

            if instruc.split(' ')[0] == 'c_if':
                gate = instruc.split(' ')[1]

                # Error propagation from result
                if measurement_readout[qubits_index[0]]: 
                    qubits[qubits_index[1]].SingleQubitGate(gate)

                # Apply gate
                prob *= 1/2
                if error != 'No gate':
                    if error == 'None':
                        qubits[qubits_index[1]].SingleQubitGate(gate)
                        prob *= (1 - var)
                    else:
                        qubits[qubits_index[1]].SingleQubitGate(error)
                        prob *= (1/4)*var
                else:
                    # add 1/2 to prob
                    pass

        if stabilizer is not None:
            is_commute = True
            for stab, q_index in zip(stabilizer, index_stabilizer):
                if stab != 'I' and measurement_readout[q_index]:
                    is_commute = not is_commute
            return measurement_readout, prob, is_commute

        return measurement_readout, prob    

    def DFE(self, Stabilizers, IndexStabilizer, Conditions, measure_error_var='p_meas'):

        Expectation_value = {
            stabilizer: {
                'commute': 0,
                'anti-commute': 0
            }
        for stabilizer in Stabilizers}

        # Add measure in basis corresponding to stabilzer
        for stabilizer in tqdm.tqdm(Stabilizers, desc='Stabilizer'):
            
            meas_var = Symbol(measure_error_var) if isinstance(measure_error_var, str) else measure_error_var
            # Copy instruction
            instructions = self.instructions[:]


            for stab, qubit_index in zip(stabilizer, IndexStabilizer):
                instructions.append({
                    'instruction': 'measure ' + stab,
                    'qubits' : qubit_index, 
                    'prob_var': meas_var,
                    'variation': ['None', 'Flip'] if stab != 'I' or meas_var != 0 else ['None']
                })

            # Evalulate instruction
            variations = [instruc['variation'] for instruc in instructions]
            for error in tqdm.tqdm(itertools.product(*variations), desc='Variations'):
                measurement_readout, prob, is_commute = self.evalute(error, instructions, stabilizer, IndexStabilizer)
            
                #print( error, measurement_readout, is_commute, stabilizer, prob )

                pass_condition = True
                if Conditions is not None:
                    measurement_results = []
                    for condition in Conditions:
                        measurement_results.append([
                            measurement_readout[q] for q in condition
                        ])

                    # if condition is satisfy use result to calculate fidleity
                    if not all( [ True if len(set(i)) == 1 else False for i in measurement_results ] ):
                        pass_condition = False
                
                if pass_condition:
                    if is_commute:
                        Expectation_value[stabilizer]['commute'] += prob
                    else:
                        Expectation_value[stabilizer]['anti-commute'] += prob

        summation = 0
        for stabilizer in Expectation_value:
            summation += Expectation_value[stabilizer]['commute']
            summation -= Expectation_value[stabilizer]['anti-commute']
        
        Fidelity = (1/(len(Stabilizers) + 1)) * (1 + summation) 

        # return Expectation_value

        # print(Expectation_value)

        return Fidelity
                