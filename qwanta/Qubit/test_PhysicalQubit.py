import unittest
from qubit import PhysicalQubit


class VirtualSimpyEnvironment:
    def __init__(self) -> None:
        self.now = 0

class TestPhysicalQubit(unittest.TestCase):

    def setUp(self) -> None:
        self.env = VirtualSimpyEnvironment()
        self.memFunc = [1, 0, 0, 0]
        self.gateError = 0
        self.measurementError = 0
        self.qubit = PhysicalQubit('EndNode1', 'EndNode2', 0, 'EndNode1-EndNode2', 'test_qubit', self.env, 'test', self.memFunc, self.gateError, self.measurementError)
        self.qubit.setInitialTime()
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_HadamardGate(self):
        self.qubit.error_x = True 
        self.qubit.H_gate()
        self.assertEqual(self.qubit.error_z, True)
        self.assertEqual(self.qubit.error_x, False)

    def test_Xgate(self):
        self.qubit.X_gate()
        self.assertEqual(self.qubit.error_x, True)

    def test_Zgate(self):
        self.qubit.Z_gate()
        self.assertEqual(self.qubit.error_z, True)

    def test_CNOTgate(self):
        control_qubit = PhysicalQubit('EndNode1', 'EndNode2', 1, 'EndNode1-EndNode2', 'test_target_qubit', self.env, 'test', self.memFunc, self.gateError, self.measurementError)
        control_qubit.error_x = True

        # X error propagatation
        self.qubit.CNOT_gate(control_qubit)
        self.assertEqual(self.qubit.error_x, True)

        # Z error propagation
        self.qubit.error_z = True
        self.qubit.CNOT_gate(control_qubit)
        self.assertEqual(control_qubit.error_z, True)

    def test_measureZ(self):

        result = self.qubit.measureZ()
        self.assertEqual(result, False)

        self.qubit.error_x = True 
        result = self.qubit.measureZ()
        self.assertEqual(result, True)

    def test_measureX(self):

        result = self.qubit.measureX()
        self.assertEqual(result, False)

        self.qubit.error_z = True 
        result = self.qubit.measureX()
        self.assertEqual(result, True)

    def test_measureY(self):

        result = self.qubit.measureY()
        self.assertEqual(result, False)

        self.qubit.error_z = True
        result = self.qubit.measureY()
        self.assertEqual(result, True)
        self.qubit.error_z = False

        self.qubit.error_x = True
        result = self.qubit.measureY()
        self.assertEqual(result, True)
        self.qubit.error_x = False

        self.qubit.error_z = True
        self.qubit.error_x = True
        result = self.qubit.measureY()
        self.assertEqual(result, False)

