import simpy 
import random
import numpy as np
from geopy.distance import distance, great_circle

class Mixin:

    def getDistance(self, node1, node2, t=None):

        time = self.env.now if t is None else t

        # Get coordinate of node1 and node2
        point1 = self.configuration.nodes_info[node1]['coordinate']
        point2 = self.configuration.nodes_info[node2]['coordinate']

        coor1 = point1(time) if callable(point1) else point1
        coor2 = point2(time) if callable(point2) else point2

        # Calculate eucidian distance
        #relative_distance = np.linalg.norm(np.array(coor1) - np.array(coor2))

        if self.configuration.coor_system == 'normal':
            relative_distance = np.linalg.norm(np.array(coor1) - np.array(coor2))

        else:
            if coor1[2] == 0 and coor2[2] == 0:
                relative_distance = great_circle(coor1[:2], coor2[:2]).km
            else:
                relative_distance = np.sqrt( (distance(coor1[:2], coor2[:2]).km)**2 + (coor1[2] - coor2[2])**2)
        
        return relative_distance

    def calculateErrorProb(self, node1, node2):

        relative_distance = self.getDistance(node1, node2)

        # Calculate probability distribution from distance of given edge
        prob = self.graph[node1][node2]['depolarlizing error'](relative_distance, self.env.now) if callable(self.graph[node1][node2]['depolarlizing error']) else self.graph[node1][node2]['depolarlizing error']

        return prob

    def getPhotonLossProb(self, node1, node2):

        relative_distance = self.getDistance(node1, node2)
        loss = self.graph[node1][node2]['loss']

        # Calculate photon loss prob based on distance and revalent information of that edge
        # dB is -10log(I/I0) -> I/I0 is prob ?
        prob = 1/(10**(loss*relative_distance/10))

        return prob

    def condition(self, node1, node2):

        # Create elementary link if condition met

        return True

    def getTimeToTravel(self, node1, node2, d=None):

        # Use distance between node 1 and node 2 if d is not provided.
        distance = self.getDistance(node1, node2) if d is None else d

        time = distance / self.graph[node1][node2]['light speed']

        return time

    def generatePhysicalResource(self, node1, node2, label_out='_Physical', num_required=1):

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        isSuccess = 0

        # classical notification for emittion process
        yield self.env.process(self.classicalCommunication(node1, node2))

        while isSuccess < num_required: 

            if self.condition(node1, node2):

                # get free resource
                event = yield simpy.AllOf(self.env, [self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].get(), 
                                                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].get()])
        
                freeQubitNode1 = yield event.events[0]
                freeQubitNode2 = yield event.events[1]

                # Perform check
                if freeQubitNode1.qubit_node_address != node1 or freeQubitNode2.qubit_node_address != node2:
                    raise ValueError("Wrong qubit address.")

                # Set initial time
                if freeQubitNode1.initiateTime is not None or freeQubitNode2.initiateTime is not None:
                    raise ValueError('This qubit has not set free properly')

                # Delay initilization ?
                freeQubitNode1.setInitialTime()
                
                # TODO
                # Pre-calculate photon loss probability
                prob = self.getPhotonLossProb(node1, node2)

                # Emit photon
                yield self.env.process(self.photonTravelingProcess(node1, node2))

                freeQubitNode2.setInitialTime()

                # classical notification for result
                yield self.env.process(self.classicalCommunication(node1, node2))

                # Random if get the resource or not
                x = random.random()
                #if x < self.configuration.BSAsuccessRate: # <= to be replace with prob
                if x < prob:

                    error_dis = self.calculateErrorProb(node1, node2)

                    #freeQubitNode1.applySingleQubitGateError(prob=self.configuration.photonChannel) # <= to be replace with error_dis
                    #freeQubitNode2.applySingleQubitGateError(prob=self.configuration.photonChannel) # <= to be replace with error_dis

                    freeQubitNode1.applySingleQubitGateError(prob=error_dis) # <= to be replace with error_dis
                    freeQubitNode2.applySingleQubitGateError(prob=error_dis) # <= to be replace with error_dis
                    
                    # Update resource and set busy to true
                    self.createLinkResource(node1, node2, freeQubitNode1, freeQubitNode2, self.resourceTables['physicalResourceTable'], label=label_out, initial=True)
                    
                    if num_required is not True:
                        isSuccess += 1
                else:
                    freeQubitNode1.setFree(); freeQubitNode2.setFree()
                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].put(freeQubitNode1)
                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].put(freeQubitNode2)


    def generatePhysicalResourceEPPS(self, node1, node2, middleNode, label_out='_Physical', num_required=1):

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

                # Perform check
                if freeQubitNode1.qubit_node_address != node1 or freeQubitNode2.qubit_node_address != node2:
                    raise ValueError("Wrong qubit address.")

                # Set initial time
                if freeQubitNode1.initiateTime is not None or freeQubitNode2.initiateTime is not None:
                    raise ValueError('This qubit has not set free properly')
                freeQubitNode1.setInitialTime()
                freeQubitNode2.setInitialTime()
                
                prob_to_node1 = self.getPhotonLossProb(middleNode, node1)
                prob_to_node2 = self.getPhotonLossProb(middleNode, node2)

                distance_to_node1 = self.getDistance(middleNode, node1)
                distance_to_node2 = self.getDistance(middleNode, node2)

                longer_node = node1 if distance_to_node1 > distance_to_node2 else node2

                yield self.env.process(self.photonTravelingProcess(middleNode, longer_node))

                # classical notification for result
                yield self.env.process(self.classicalCommunication(middleNode, longer_node))

                x = random.random()
                if x < prob_to_node1*prob_to_node2:
                    
                    error_dis_node1 = self.calculateErrorProb(middleNode, node1)
                    error_dis_node2 = self.calculateErrorProb(middleNode, node2)

                    freeQubitNode1.applySingleQubitGateError(prob=error_dis_node1)
                    freeQubitNode2.applySingleQubitGateError(prob=error_dis_node2)
                    
                    # Update resource and set busy to true
                    self.createLinkResource(node1, node2, freeQubitNode1, freeQubitNode2, self.resourceTables['physicalResourceTable'], label=label_out, initial=True)
                    
                    if num_required is not True:
                        isSuccess += 1
                else:
                    # Set free to reset initial time.
                    freeQubitNode1.setFree(); freeQubitNode2.setFree()
                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].put(freeQubitNode1)
                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].put(freeQubitNode2)

    def generatePhysicalResourceEPPSPulse(self, node1, node2, middleNode, label_out='_Physical', num_required=1):

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        isSuccess = 0
        First_Pulse = True

        while isSuccess < num_required: 

            if self.condition(node1, node2):

                # get free resource
                event = yield simpy.AllOf(self.env, [self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].get(), 
                                                     self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].get()])
        
                freeQubitNode1 = yield event.events[0]
                freeQubitNode2 = yield event.events[1]

                # Perform check
                if freeQubitNode1.qubit_node_address != node1 or freeQubitNode2.qubit_node_address != node2:
                    raise ValueError("Wrong qubit address.")
                
                prob_to_node1 = self.getPhotonLossProb(middleNode, node1)
                prob_to_node2 = self.getPhotonLossProb(middleNode, node2)

                distance_to_node1 = self.getDistance(middleNode, node1)
                distance_to_node2 = self.getDistance(middleNode, node2)

                longer_node = node1 if distance_to_node1 > distance_to_node2 else node2

                if First_Pulse:
                    # Emit photon
                    yield self.env.process(self.photonTravelingProcess(middleNode, longer_node))

                    # classical notification for result
                    yield self.env.process(self.classicalCommunication(middleNode, longer_node))

                    First_Pulse = False
                else:
                    yield self.env.timeout(0.00001) # Pulse rate

                x = random.random()
                if x < prob_to_node1*prob_to_node2:
                    
                    error_dis_node1 = self.calculateErrorProb(middleNode, node1)
                    error_dis_node2 = self.calculateErrorProb(middleNode, node2)

                    freeQubitNode1.applySingleQubitGateError(prob=error_dis_node1)
                    freeQubitNode2.applySingleQubitGateError(prob=error_dis_node2)
                    
                    # Update resource and set busy to true
                    self.createLinkResource(node1, node2, freeQubitNode1, freeQubitNode2, self.resourceTables['physicalResourceTable'], label=label_out, initial=True)
                    
                    if num_required is not True:
                        isSuccess += 1
                else:
                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].put(freeQubitNode1)
                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].put(freeQubitNode2)

    def generatePhysicalResourceStaticPulse(self, node1, node2, label_out='_Physical', num_required=1):

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        isSuccess = 0
        First_Pulse = True

        # classical notification for emittion process
        yield self.env.process(self.classicalCommunication(node1, node2))

        while isSuccess < num_required: 

            if self.condition(node1, node2):

                # get free resource
                event = yield simpy.AllOf(self.env, [self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].get(), 
                                                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].get()])
        
                freeQubitNode1 = yield event.events[0]
                freeQubitNode2 = yield event.events[1]

                # Perform check
                if freeQubitNode1.qubit_node_address != node1 or freeQubitNode2.qubit_node_address != node2:
                    raise ValueError("Wrong qubit address.")

                # self.updateLog({'Time': self.env.now, 'Message': f'get qubit in {node1} {node2}'})

                # Pre-calculate photon loss probability
                prob = self.getPhotonLossProb(node1, node2)

                if First_Pulse:
                    # Emit photon
                    yield self.env.process(self.photonTravelingProcess(node1, node2))

                    # classical notification for result
                    yield self.env.process(self.classicalCommunication(node1, node2))

                    First_Pulse = False
                else:
                    yield self.env.timeout(0.00001) # Pulse rate

                # Random if get the resource or not
                x = random.random()
                if x < prob:

                    error_dis = self.calculateErrorProb(node1, node2)

                    freeQubitNode1.applySingleQubitGateError(prob=error_dis)
                    freeQubitNode2.applySingleQubitGateError(prob=error_dis)
                
                    # Update resource and set busy to true
                    self.createLinkResource(node1, node2, freeQubitNode1, freeQubitNode2, self.resourceTables['physicalResourceTable'], label=label_out, initial=True)
                    
                    if num_required is not True:
                        isSuccess += 1
                else:
                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].put(freeQubitNode1)
                    self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].put(freeQubitNode2)

    def Emitter(self, node1, node2, label_out='_Physical', num_required=1, middleNode=None, EPPS=False):

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

                self.env.process(self.PhotonTraveling(node1, node2, freeQubitNode1, freeQubitNode2, middleNode, EPPS))

                # Add delay before next emittion
                pulse_rate = self.graph[node1][node2]['Pulse rate'] 
                yield self.env.timeout(pulse_rate) 
                self.numBaseBellAttempt += 1

    def PhotonTraveling(self, node1, node2, qubit1, qubit2, middleNode=None, EPPS=False):

        # Many MAGICS could happen here!

        if middleNode is not None and (not EPPS):

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

        elif middleNode is not None and EPPS:

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

    def Detector(self, node1, node2, label_out='_Physical', num_required=1, middleNode=None, EPPS=False):

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

    def ClassicalMessageTraveling(self, node1, node2, message, middleNode=None, EPPS=False):

        # Processing unit lagging should be here

        if middleNode is not None and (not EPPS):
            distance_to_node1 = self.getDistance(middleNode, node1)
            distance_to_node2 = self.getDistance(middleNode, node2)
            longer_node = node1 if distance_to_node1 > distance_to_node2 else node2
            travel_time = self.getTimeToTravel(longer_node, middleNode)

        elif middleNode is not None and EPPS:
            travel_time = self.getTimeToTravel(node1, node2)
        else:
            travel_time = self.getTimeToTravel(node1, node2)

        yield self.env.timeout(travel_time)
        self.ClassicalChannel[f'{node1}-{node2}'].put(message)

    def ClassicalMessageHandler(self, node1, node2, label_out='_Physical', num_required=1, middleNode=None):

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
            
                # Update resource and set busy to true
                self.createLinkResource(node1, node2, freeQubitNode1, freeQubitNode2, self.resourceTables['physicalResourceTable'], label=label_out, initial=True)
                
                if num_required is not True:
                    isSuccess += 1
            else:
                # Set free to reset initial time. 
                # Since classical message is herald, We assume that information about returning qubit to table is also transfer
                freeQubitNode1.setFree(); freeQubitNode2.setFree()
                self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node1}'].put(freeQubitNode1)
                self.QubitsTables['externalQubitsTable'][f'{node1}-{node2}'][f'QNICs-{node2}'].put(freeQubitNode2)