import simpy 
import random
import numpy as np
from geopy.distance import distance, great_circle
from ..Qubit import PhysicalQubit
from typing import Union, Any, Optional, Tuple

class Mixin:

    def getDistance(self, node1: Any, node2: Any, t: Optional[Union[float, int]]=None):
        """Calculate distance between `node1` and `node2`

        Args:
            node1 (Any): node 1
            node2 (Any): node 2
            t (Optional[Union[float, int]], optional): time at the moment of function call. Defaults to None.

        Returns:
            float: relative distance between node 1 and node 2
        """

        time = self.env.now if t is None else t

        # Get coordinate of node1 and node2
        point1 = self.configuration.nodes_info[node1]['coordinate']
        point2 = self.configuration.nodes_info[node2]['coordinate']

        coor1 = point1(time) if callable(point1) else point1
        coor2 = point2(time) if callable(point2) else point2

        # Calculate eucidian distance,
        if self.configuration.coor_system == 'normal':
            relative_distance = np.linalg.norm(np.array(coor1) - np.array(coor2))

        # Calculate distance according to eacth curvature.
        else:
            # If it is on the ground 
            if coor1[2] == 0 and coor2[2] == 0:
                relative_distance = great_circle(coor1[:2], coor2[:2]).km
            # If it is 3-dimension coordinate.
            else:
                relative_distance = np.sqrt( (distance(coor1[:2], coor2[:2]).km)**2 + (coor1[2] - coor2[2])**2)
        
        return relative_distance

    def calculateErrorProb(self, node1: Any, node2: Any):
        """Calculate depolarizing error probablity depend on edge infomation.

        Args:
            node1 (Any): node 1
            node2 (Any): node 2

        Returns:
            float: probablity of depolarizing channel
        """

        relative_distance = self.getDistance(node1, node2)

        # Calculate probability distribution from distance of given edge
        prob = self.graph[node1][node2]['depolarlizing error'](relative_distance, self.env.now) if callable(self.graph[node1][node2]['depolarlizing error']) else self.graph[node1][node2]['depolarlizing error']

        return prob

    def getPhotonLossProb(self, node1: Any, node2: Any):
        """Calculate probablity of photon arriving at detector and deterctor detect the photon.

        Args:
            node1 (Any): node 1
            node2 (Any): node 2

        Returns:
            float: probablity of photon arriving at detector and deterctor detect the photon
        """

        relative_distance = self.getDistance(node1, node2)
        loss = self.graph[node1][node2]['loss']

        # Calculate probablity of photon arriving at detector and deterctor detect the photon 
        # based on distance and revalent information of that edge
        # dB is -10log(I/I0) -> I/I0 is prob ?
        
        prob = 1/(10**(loss*relative_distance/10))

        return prob

    def condition(self, node1: Any, node2: Any):
        """Function to check if condtion is meet between `node1` and `node2` or not.
           Currently, function always return `True`

        Args:
            node1 (Any): node 1
            node2 (Any): node 2

        Returns:
            bool: return if condition is met.
        """

        # Create elementary link if condition met

        return True

    def getTimeToTravel(self, node1:Any, node2:Any, d:Optional[Union[float, int]]=None):
        """Calculate time required to travel from `node1` to `node2`

        Args:
            node1 (Any): node 1
            node2 (Any): node 2
            d (Optional[Union[float, int]], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        # Use distance between node 1 and node 2 if d is not provided.
        distance = self.getDistance(node1, node2) if d is None else d

        time = distance / self.graph[node1][node2]['light speed']

        return time

    def Emitter(self, 
                node1: Any, 
                node2: Any, 
                label_out: Optional[str] = '_Physical', 
                num_required: Optional[int] = 1, 
                middleNode: Optional[Any] = None, 
                EPPS: Optional[bool] = False):
        """Process for getting free physical qubit and try to establish link level entanglement. 

        Args:
            node1 (Any): node 1 that the process is executed.
            node2 (Any): node 2 that the process is executed.
            label_out (Optional[str], optional): Output label of resource. Defaults to '_Physical'.
            num_required (Optional[int], optional): Number of time required to execute. Defaults to 1.
            middleNode (Optional[Any], optional): middle node. Defaults to None.
            EPPS (Optional[bool], optional): Whether this is proces for EPPS model or not. Defaults to False.

        Raises:
            ValueError: Qubit is not set free before used.

        Yields:
            _type_: _description_
        """

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        isSuccess = 0

        while isSuccess < num_required: 

            if self.condition(node1, node2):

                # get free resource
                event = yield simpy.AllOf(self.env, [self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].get(), 
                                                     self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].get()])
        
                freeQubitNode1 = yield event.events[0]
                freeQubitNode2 = yield event.events[1]

                if freeQubitNode1.initiateTime is not None or freeQubitNode2.initiateTime is not None:
                    raise ValueError('This qubit has not set free properly')

                # Initiate subsequnce process for traveling of photon.
                self.env.process(self.PhotonTraveling(node1, node2, freeQubitNode1, freeQubitNode2, middleNode, EPPS))

                # Add delay before next emittion
                pulse_rate = self.graph[node1][node2]['Pulse rate'] 
                yield self.env.timeout(pulse_rate) 
                self.numBaseBellAttempt += 1

    def PhotonTraveling(self, 
                        node1: Any, 
                        node2: Any, 
                        qubit1: PhysicalQubit, 
                        qubit2: PhysicalQubit, 
                        middleNode: Optional[Any] = None, 
                        EPPS: Optional[bool] = False):
        """This process will induce a time delay nescessary for each model,
           initial matter qubit when needed, calculate probability of photon arrival and sent to `Detector` process.

        Args:
            node1 (Any): node 1
            node2 (Any): node 2
            qubit1 (PhysicalQubit): qubit of node 1
            qubit2 (PhysicalQubit): qubit of node 2
            middleNode (Optional[Any], optional): middle node. Defaults to None.
            EPPS (Optional[bool], optional): Whether this is proces for EPPS model or not. Defaults to False.

        Yields:
            _type_: _description_
        """

        # Many MAGICS could happen here!

        # For BSA in the middle model
        if (middleNode != None) and (not EPPS):

            node1_to_middle_time = self.getTimeToTravel(node1, middleNode)
            node2_to_middle_time = self.getTimeToTravel(node2, middleNode)

            # Delay for some time before the time used to travel to middle node is the same then initialize the nearer qubit
            delay_time = abs(node1_to_middle_time - node2_to_middle_time)
            if node1_to_middle_time < node2_to_middle_time:
                # The time needed to travel from node 1 to middle is shorter than node 2 to middle
                qubit1.setInitialTime()
                yield self.env.timeout(delay_time)
                qubit2.setInitialTime()
            else:
                # Otherwise
                qubit2.setInitialTime()
                yield self.env.timeout(delay_time)
                qubit1.setInitialTime()
            
            shorter_time = node1_to_middle_time if node1_to_middle_time < node2_to_middle_time else node2_to_middle_time

            prob = self.getPhotonLossProb(node1, middleNode) * self.getPhotonLossProb(middleNode, node2)
            yield self.env.timeout(shorter_time)

        # For EPPS model
        elif (middleNode != None) and EPPS:

            middle_to_node1_time = self.getTimeToTravel(node1, middleNode)
            middle_to_node2_time = self.getTimeToTravel(node2, middleNode)

            if middle_to_node1_time < middle_to_node2_time:
                # The time needed to travel from node 1 to middle is shorter than node 2 to middle
                
                yield self.env.timeout(middle_to_node1_time)
                qubit1.setInitialTime()

                yield self.env.timeout(middle_to_node2_time - middle_to_node1_time)
                qubit2.setInitialTime()
            else:
                # Otherwise
                yield self.env.timeout(middle_to_node2_time)
                qubit1.setInitialTime()

                yield self.env.timeout(middle_to_node1_time - middle_to_node2_time)
                qubit2.setInitialTime()

            prob = self.getPhotonLossProb(node1, middleNode) * self.getPhotonLossProb(middleNode, node2)

        # For sender-reciever model
        else:
            # initial qubit on the left
            qubit1.setInitialTime()
            travel_time = self.getTimeToTravel(node1, node2)
            
            # Determine success or fail
            # Pre-calculate photon loss probability
            prob = self.getPhotonLossProb(node1, node2)

            yield self.env.timeout(travel_time)
            qubit2.setInitialTime()
        self.QuantumChannel[f'{node1}-{node2}'].put((qubit1, qubit2, prob))

    def Detector(self, 
                 node1: Any, 
                 node2: Any, 
                 label_out: Optional[str] = '_Physical', 
                 num_required: Optional[int] = 1, 
                 middleNode: Optional[Any] = None, 
                 EPPS: Optional[bool] = False):
        """Detector process initial on edge of `node1` and `node2`
           This process random a float from uniform distribution to decide the link-connection result.

        Args:
            node1 (Any): node 1 that the process is executed.
            node2 (Any): node 2 that the process is executed.
            label_out (Optional[str], optional): Output label of resource. Defaults to '_Physical'.
            num_required (Optional[int], optional): Number of time required to execute. Defaults to 1.
            middleNode (Optional[Any], optional): middle node. Defaults to None.
            EPPS (Optional[bool], optional): Whether this is proces for EPPS model or not. Defaults to False.

        Yields:
            _type_: _description_
        """

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        isSuccess = 0

        while isSuccess < num_required: 

            if self.condition(node1, node2):

                # Get photon qubit from both node when ever they are here
                qubit1, qubit2, prob = yield self.QuantumChannel[f'{node1}-{node2}'].get()

                x = random.random()
                if x < prob:
                    # Success

                    # Message can be extend to class 
                    message = (qubit1, qubit2, True)

                else:
                    # Fail
                    message = (qubit1, qubit2, False)
                
                self.env.process(self.ClassicalMessageTraveling(node1, node2, message, middleNode, EPPS))

    def ClassicalMessageTraveling(self, 
                                  node1: Any, 
                                  node2: Any, 
                                  message: Tuple, 
                                  middleNode: Optional[Any] = None, 
                                  EPPS: Optional[bool] = False):
        """Depend on model, time delay needed for result will be invoked.
           Then the process will put the message in the classical channel

        Args:
            node1 (Any): node 1
            node2 (Any): Node 2
            message (Tuple): Message to be sent from node 2 to node 1
            middleNode (Optional[Any], optional): middle node. Defaults to None.
            EPPS (Optional[bool], optional): Whether this is proces for EPPS model or not. Defaults to False.

        Yields:
            _type_: _description_
        """

        # Processing unit lagging should be here

        # For BSA in the middle, induce time delay needed for longer node only.
        if (middleNode != None) and (not EPPS):
            distance_to_node1 = self.getDistance(middleNode, node1)
            distance_to_node2 = self.getDistance(middleNode, node2)
            longer_node = node1 if distance_to_node1 > distance_to_node2 else node2
            travel_time = self.getTimeToTravel(longer_node, middleNode)

        # For EPPS model, induce time delay needed for traveling from node 1 to node 2.
        elif (middleNode != None) and EPPS:
            travel_time = self.getTimeToTravel(node1, node2)
        # For sender-reciever model, induce time delay needed for traveling from node 1 to node 2.
        else:
            travel_time = self.getTimeToTravel(node1, node2)

        yield self.env.timeout(travel_time)
        self.ClassicalChannel[f'{node1}-{node2}'].put(message)

    def ClassicalMessageHandler(self, 
                                node1: Any, 
                                node2: Any, 
                                label_out: Optional[str] = '_Physical', 
                                num_required: Optional[int] = 1, 
                                middleNode: Optional[Any] = None):
        """Check the result, 
                If True, then apply depolarizing error to the matter qubits on each node and register the new resource.
                If False, Set qubit free and return qubit to the self.QubitTables, as both nodes know that result is failed
                          then no delay time is needed to induced to notify thier neighbor node.
           

        Args:
            node1 (Any): _description_
            node2 (Any): _description_
            label_out (Optional[str], optional): _description_. Defaults to '_Physical'.
            num_required (Optional[int], optional): _description_. Defaults to 1.
            middleNode (Optional[Any], optional): _description_. Defaults to None.

        Yields:
            _type_: _description_
        """

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        isSuccess = 0

        while isSuccess < num_required: 

            message = yield self.ClassicalChannel[f'{node1}-{node2}'].get()

            freeQubitNode1, freeQubitNode2, result = message

            if result:
                error_dis = self.calculateErrorProb(node1, node2)

                freeQubitNode1.applySingleQubitGateError(prob=error_dis)
                freeQubitNode2.applySingleQubitGateError(prob=error_dis)
            
                # Update resource
                self.createLinkResource(node1, node2, freeQubitNode1, freeQubitNode2, 'physicalResourceTable', label=label_out)
                
                if not isinstance(num_required, bool):
                    isSuccess += 1
            else:
                # Set free to reset initial time. 
                # Since classical message is herald, We assume that information about returning qubit to table is also transfer
                freeQubitNode1.setFree(); freeQubitNode2.setFree()
                self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].put(freeQubitNode1)
                self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].put(freeQubitNode2)