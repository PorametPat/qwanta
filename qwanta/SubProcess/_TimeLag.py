import networkx as nx

class Mixin:

    '''
    Time lag process
    '''
    def ConnectionSetup(self, initiator, responder):

        # Send connection setup from initiator to responder
        self.updateLog({'Time': self.env.now, 'Message': f'Send connection request from {initiator} to {responder}'})
        yield self.env.process(self.classicalCommunication(initiator, responder))

        # Return RuleSet
        self.updateLog({'Time': self.env.now, 'Message': f'Return RuleSet request from {responder} to {initiator}'})
        yield self.env.process(self.classicalCommunication(responder, initiator))
        self.updateLog({'Time': self.env.now, 'Message': f'Process RuleSet of {initiator} and {responder}'})

        # Initialize time stamp after connection setup
        self.connectionSetupTimeStamp = self.env.now

    def classicalCommunication(self, source, destination, factor=1):
        # Calculate time from distance, in this case simply using edges as a traveling time 
        # self.env.timeout(0)

        path = nx.dijkstra_path(self.graph, source, destination)
        for i in range(len(path)-1):
            for u, v, w in self.graph.edges(data=True):
                w['weight'] = self.getDistance(u, v) / self.graph[u][v]['light speed']
            yield self.env.timeout(factor*nx.dijkstra_path_length(self.graph, path[i], path[i+1]))

        '''
        for i in range(len(path)-1):
            for u, v, w in self.graph.edges(data=True):
                w['weight'] = self.getDistance(u, v) / self.graph[u][v]['light speed']
            self.graph.edges()
            yield self.env.timeout(factor*nx.dijkstra_path_length(self.graph, path[i], path[i+1]))
        '''

        '''
        # Recalculate time to travel
        for u, v, w in self.graph.edges(data=True):
            w['weight'] = self.getDistance(u, v) / self.graph[u][v]['light speed']

        #yield self.env.timeout(self.getTimeToTravel(source, destination))
        yield self.env.timeout(factor*nx.dijkstra_path_length(self.graph, source, destination))
        '''

    def photonTravelingProcess(self, source, destination):
        # Calculate time from distance, in this case simply using edges as a traveling time
        # self.env.timeout(1)
        path = nx.dijkstra_path(self.graph, source, destination)
        for i in range(len(path)-1):
            for u, v, w in self.graph.edges(data=True):
                w['weight'] = self.getDistance(u, v) / self.graph[u][v]['light speed']
            yield self.env.timeout(nx.dijkstra_path_length(self.graph, path[i], path[i+1]))

        '''
        # Recalculate time to travel
        for u, v, w in self.graph.edges(data=True):
            w['weight'] = self.getDistance(u, v) / self.graph[u][v]['light speed']

        #yield self.env.timeout(self.getTimeToTravel(source, destination))
        yield self.env.timeout(nx.dijkstra_path_length(self.graph, source, destination))
        '''