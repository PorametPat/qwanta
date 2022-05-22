import unittest
from qubit import LogicalQubit, PhysicalQubit

class VirtualSimpyEnvironment:
    def __init__(self) -> None:
        self.now = 0

class TestLogicalQubit(unittest.TestCase):

    def setUp(self) -> None:
        self.env = VirtualSimpyEnvironment()
        self.memFunc = [1, 0, 0, 0]
        self.gateError = 0
        self.measurementError = 0
        PhysicalQubits = [PhysicalQubit('Damm', 'you', i, 'boy!', 'encoding_qubit', self.env, 'test', self.memFunc, self.gateError, self.measurementError) for i in range(7)]
        self.logicalQubit = LogicalQubit('Damm', 'Steane', 'girl!', self.env)
        self.logicalQubit.physical_list = PhysicalQubits
        for qubit in self.logicalQubit.physical_list:
            qubit.setInitialTime()
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_SteaneEncoderDecoder_Zbasis(self):

        self.logicalQubit.physical_list[0].error_x = True
        self.logicalQubit.encode()

        self.logicalQubit.decode()

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [2]:
                self.assertEqual(qubit.error_x, True)
            else:
                self.assertEqual(qubit.error_x, False)
    
    def test_measure_getOperators(self):
        # test for perfect get all operators from each physical qubits
        result = self.logicalQubit.measureForFidelity(method='get_operator')
        self.assertEqual(result, [0 for _ in range(7)])

    def test_measure_getLogicalOperator(self):
        # Using perfect decode to get logical operator
        result = self.logicalQubit.measureForFidelity(method='get_logicalOperator')
        self.assertEqual(result, 'II')

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [0, 1, 2, 4]:
                qubit.error_x = True
        result = self.logicalQubit.measureForFidelity(method='get_logicalOperator')
        self.assertEqual(result, 'XI')

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [0, 1, 2, 4]:
                qubit.error_z = True
        result = self.logicalQubit.measureForFidelity(method='get_logicalOperator')
        self.assertEqual(result, 'XZ')

    def test_measure_steane_standard_correctable_error_Z(self):

        for i in range(7):
            self.logicalQubit.physical_list[i].error_x = True

            result, location = self.logicalQubit.measure('Z', return_mode=True)
            self.assertEqual(location, i)   
            self.assertEqual(result, [False]*7)    
            
            self.logicalQubit.physical_list[i].error_x = False
    
    def test_measure_steane_standard_correctable_error_X(self):

        for i in range(7):
            self.logicalQubit.physical_list[i].error_z = True

            result, location = self.logicalQubit.measure('X', return_mode=True)
            self.assertEqual(result, [False]*7)    
            self.assertEqual(location, i)  

            self.logicalQubit.physical_list[i].error_z = False
    
    def test_measure_steane_standard_correctable_error_Y(self):

        for i in range(7):
            self.logicalQubit.physical_list[i].error_z = True

            result, location = self.logicalQubit.measure('Y', return_mode=True)
            self.assertEqual(result, [False]*7)    
            self.assertEqual(location, i)  

            self.logicalQubit.physical_list[i].error_z = False

        for i in range(7):
            self.logicalQubit.physical_list[i].error_x = True

            result, location = self.logicalQubit.measure('Y', return_mode=True)
            self.assertEqual(result, [False]*7)    
            self.assertEqual(location, i)  

            self.logicalQubit.physical_list[i].error_x = False
        
        for i in range(7):
            self.logicalQubit.physical_list[i].error_z = True
            self.logicalQubit.physical_list[i].error_x = True

            result, location = self.logicalQubit.measure('Y', return_mode=True)
            for qubit in self.logicalQubit.physical_list:
                res = qubit.measureY()
                self.assertEqual(res, False)  
            self.assertEqual(result, [False]*7)    
            self.assertEqual(location, -1)  

            self.logicalQubit.physical_list[i].error_z = False
            self.logicalQubit.physical_list[i].error_x = False
    
    def test_measure_steane_standard_non_correctable_error_Z(self):

        
        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [0, 1, 2, 6]:
                qubit.error_x = True

        result, location = self.logicalQubit.measure('Z', return_mode=True)
        self.assertEqual(result, [True, True, True, False, False, False, False])    
        self.assertEqual(location, 6)    

    def test_measure_steane_standard_correctable_error_Z_2(self):

        for i in range(7):
            self.logicalQubit.physical_list[i].error_x = True

            result, location = self.logicalQubit.measureZ(return_mode=True)
            self.assertEqual(location, i)   
            self.assertEqual(result, [False]*7)    
            
            self.logicalQubit.physical_list[i].error_x = False
    
    def test_measure_steane_standard_correctable_error_X_2(self):

        for i in range(7):
            self.logicalQubit.physical_list[i].error_z = True

            result, location = self.logicalQubit.measureX(return_mode=True)
            self.assertEqual(result, [False]*7)    
            self.assertEqual(location, i)  

            self.logicalQubit.physical_list[i].error_z = False
    
    def test_measure_steane_standard_correctable_error_Y_2(self):

        for i in range(7):
            self.logicalQubit.physical_list[i].error_z = True

            result, location = self.logicalQubit.measureY(return_mode=True)
            self.assertEqual(result, [False]*7)    
            self.assertEqual(location, i)  

            self.logicalQubit.physical_list[i].error_z = False

        for i in range(7):
            self.logicalQubit.physical_list[i].error_x = True

            result, location = self.logicalQubit.measureY(return_mode=True)
            self.assertEqual(result, [False]*7)    
            self.assertEqual(location, i)  

            self.logicalQubit.physical_list[i].error_x = False
        
        for i in range(7):
            self.logicalQubit.physical_list[i].error_z = True
            self.logicalQubit.physical_list[i].error_x = True

            result, location = self.logicalQubit.measureY(return_mode=True)
            for qubit in self.logicalQubit.physical_list:
                res = qubit.measureY()
                self.assertEqual(res, False)  
            self.assertEqual(result, [False]*7)    
            self.assertEqual(location, -1)  

            self.logicalQubit.physical_list[i].error_z = False
            self.logicalQubit.physical_list[i].error_x = False
    
    def test_measure_steane_standard_non_correctable_error_Z_2(self):

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [0, 1, 2, 6]:
                qubit.error_x = True

        result, location = self.logicalQubit.measureZ(return_mode=True)
        self.assertEqual(result, [True, True, True, False, False, False, False])    
        self.assertEqual(location, 6)  

    def test_measure_steane_standard_non_correctable_error_Z_get_operator(self):

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [0, 1, 2, 6]:
                qubit.error_x = True
            if index in [0, 3, 4]:
                qubit.error_z = True

        result, operator = self.logicalQubit.measureZ(get_operator=True)
        self.assertEqual(result, True)    
        self.assertEqual(operator, [1, 2, 2, 3, 3, 0, 2])  

    def test_errorDetectionCorrection_steane_standard(self):

        AncillaQubits = [PhysicalQubit('Damm', 'you', i, 'boy!', 'detecting_qubit', self.env, 'test', self.memFunc, self.gateError, self.measurementError) for i in range(6)]
        for qubit in AncillaQubits:
            qubit.setInitialTime()
        self.logicalQubit.ancilla_list = AncillaQubits


        self.logicalQubit.error_detection_correction()

        for qubit in self.logicalQubit.physical_list:
            self.assertEqual(qubit.error_x, False)
            self.assertEqual(qubit.error_z, False)

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [1]:
                qubit.error_x = True
                qubit.error_z = True
        
        syn_x, syn_z = self.logicalQubit.error_detection_correction(return_syndrome=True)
        self.assertEqual(syn_x, 1)
        self.assertEqual(syn_z, 1)

        for qubit in self.logicalQubit.physical_list:
            self.assertEqual(qubit.error_x, False)
            self.assertEqual(qubit.error_z, False)

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [0, 1, 2, 6]:
                qubit.error_x = True
    
        syn_x, syn_z = self.logicalQubit.error_detection_correction(return_syndrome=True)
        self.assertEqual(syn_x, 6)
        self.assertEqual(syn_z, -1)

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [3, 4, 5, 6]:
                self.assertEqual(qubit.error_x, False)
            else:
                self.assertEqual(qubit.error_x, True)

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [0, 1, 2, 6]:
                qubit.error_z = True
    
        syn_x, syn_z = self.logicalQubit.error_detection_correction(return_syndrome=True)
        self.assertEqual(syn_x, -1)
        self.assertEqual(syn_z, 6)

        for index, qubit in enumerate(self.logicalQubit.physical_list):
            if index in [3, 4, 5, 6]:
                self.assertEqual(qubit.error_z, False)
            else:
                self.assertEqual(qubit.error_z, True)

    def test_perfect_error_correction(self):

        env = VirtualSimpyEnvironment()
        memFunc = [0.25, .25, .25, .25]
        gateError = 0.5
        measurementError = 1
        PhysicalQubits = [PhysicalQubit('Damm', 'you', i, 'boy!', 'encoding_qubit', env, 'test', memFunc, gateError, measurementError) for i in range(7)]
        logicalQubit = LogicalQubit('Damm', 'Steane', 'girl!', env)
        logicalQubit.physical_list = PhysicalQubits
        for qubit in logicalQubit.physical_list:
            qubit.setInitialTime()

        AncillaQubits = [PhysicalQubit('Damm', 'you', i, 'boy!', 'detecting_qubit', env, 'test', memFunc, gateError, measurementError) for i in range(6)]
        for qubit in AncillaQubits:
            qubit.setInitialTime()
        logicalQubit.ancilla_list = AncillaQubits

        for index, qubit in enumerate(logicalQubit.physical_list):
            
            qubit.error_x = True
            qubit.error_z = True

            logicalQubit.error_detection_correction(perfect_correction=True)

            for ii, q in enumerate(logicalQubit.physical_list):
                self.assertEqual(q.error_x, False)
                self.assertEqual(q.error_z, False)
