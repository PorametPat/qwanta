from ast import Raise
import networkx as nx
from ..Qubit import PhysicalQubit
from typing import Union, Any, Optional

class Mixin:

    '''
    Time lag process
    '''
    def ConnectionSetup(self, initiator: Any, responder: Any):
        """Process for establish setup request

        Args:
            initiator (Any): initiator node
            responder (Any): responder node

        Yields:
            _type_: _description_
        """

        # Send connection setup from initiator to responder
        self.updateLog({'Time': self.env.now, 'Message': f'Send connection request from {initiator} to {responder}'})
        yield self.env.process(self.classicalCommunication(initiator, responder))

        # Return RuleSet
        self.updateLog({'Time': self.env.now, 'Message': f'Return RuleSet request from {responder} to {initiator}'})
        yield self.env.process(self.classicalCommunication(responder, initiator))
        self.updateLog({'Time': self.env.now, 'Message': f'Process RuleSet of {initiator} and {responder}'})

        # Initialize time stamp after connection setup
        self.connectionSetupTimeStamp = self.env.now
        self.FidelityEstimationTimeStamp = self.env.now

    def classicalCommunication(self, source: Any, destination: Any, factor: Optional[Union[float, int]]=1):
        """This process will induce delay for comminucation from source to target node.

        Args:
            source (Any): source of message to be sent.
            destination (Any): reciever of message.
            factor (Optional[Union[float, int]], optional): factor to be multiply to the time required to the light speed. Defaults to 1.

        Yields:
            _type_: _description_
        """
        
        # Calculate time from distance, in this case simply using edges as a traveling time 
        path = nx.dijkstra_path(self.graph, source, destination)
        for i in range(len(path)-1):

            # Update time required to travel to adjacent ndoe of every edges
            for u, v, w in self.graph.edges(data=True):
                w['weight'] = self.getDistance(u, v) / self.graph[u][v]['light speed']
        
            # Induce the time delay
            yield self.env.timeout(factor*nx.dijkstra_path_length(self.graph, path[i], path[i+1]))

    def photonTravelingProcess(self, source: Any, destination: Any):
        """This process will induce delay for comminucation from source to target node.

        Args:
            source (Any): Source of the photon
            destination (Any): Destination node of photon

        Yields:
            _type_: _description_
        """
        
        # Calculate time from distance, in this case simply using edges as a traveling time
        path = nx.dijkstra_path(self.graph, source, destination)
        for i in range(len(path)-1):
            
            # Update time required to travel to adjacent ndoe of every edges
            for u, v, w in self.graph.edges(data=True):
                w['weight'] = self.getDistance(u, v) / self.graph[u][v]['light speed']
            
            # Induce the time delay
            yield self.env.timeout(nx.dijkstra_path_length(self.graph, path[i], path[i+1]))

    def returnToQubitTable(self, qubit: PhysicalQubit):
        """Process that induce time to send message to neighbor node of the qubit to be put back to `self.QubitTables`.

        Args:
            qubit (PhysicalQubit): physical qubit to be put to back to `self.QubitTables`.
                                   If its role is external, then delay will be induced.
                                   If its role is internal, then no delay is induced.

        Raises:
            ValueError: Invalid type of qubit's role

        Yields:
            _type_: _description_
        """
        if qubit.role == 'external':
            # Send classical message to neighbor first 
            yield self.env.process(self.classicalCommunication(qubit.qubit_node_address, qubit.qubit_neighbor_address))
        elif qubit.role == 'internal':
            # Internal notification required zero time.
            pass
        else:
            raise ValueError(f'Qubit is set with invalid type of role {qubit.role}')

        # Put resource back to its own table
        self.QubitsTables[qubit.table][qubit.qnics_address][f'QNICs-{qubit.qubit_node_address}'].put(qubit)
