import numpy as np


class Graph:
    def __init__(self, numVertices, numEdges, rates, vectors='uniform',
                 connected=1, MC=True):
        """
        :param numVertices: number of vertices = number of neurons in model,
        :param numEdges: number of edges = number of Synapses in model,
        :param rates: perturbation rates, rates = {'alpha': , 'beta': , 'epsilon': }
        :param vectors = 'uniform' or {'y': np.array([]), 'x': np.array([])},
            for deterministic equation of motion
                self.y: indegrees,
                self.x: outdegrees
            for Monte Carlo simulation
                MC: True/False, if or if no Monte Carlo simulation is going to be made
                self.edges: list of existing edges,
                        zeros and ones, one: connected, zero: disconnected,
                self.heads: corresponding list of x positions of edges,
                self.tails: corresponding list of y positions of edges,
        :param connected = 0 (completely disconnected) or 1 (completely connected)
        :param MC: whether a Monte Carlo simulation should be created or not
        """

        """Introduce all defining system parameters"""

        self.n = numVertices
        self.z = numEdges
        self.vectors = vectors
        self.connected = connected
        self.MC = MC
        self.alpha = rates['alpha']
        self.beta = rates['beta']
        self.epsilon = rates['epsilon']

        if self.vectors == 'uniform':
            self.y = np.full(self.n, self.z / self.n)
            self.x = np.copy(self.y)
            self.uniform = np.full(self.n, 1/self.n)
        else:
            self.y = self.vectors['y']
            self.x = self.vectors['x']
        if MC:
            assert (self.z / self.n) % 1 != 1, \
             'z/n={} is not an integer'.format(self.z / self.n)
            self.edges = np.full(self.z, connected)
            self.x = self.x.astype(int)
            self.y = self.y.astype(int)
            self.tails = np.repeat(np.arange(self.x.size), self.x)
            self.heads = np.repeat(np.arange(self.y.size), self.y)
            np.random.shuffle(self.heads)
