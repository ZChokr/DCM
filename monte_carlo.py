import numpy as np


class MonteCarlo:
    def __init__(self, graph, stepwidth, t_end, inh=False, shift=False):
        """
        :param graph: Graph object
        :param stepwidth: step size
        :param t_end: end of perturbation
        :param inh: whether to create excitatory and inhibitory neurons or solely excitatory ones
        :param shift: whether to shift the matrix by the intital matrix zNN^T
        """

        self.stepwidth = stepwidth
        if inh:
            assert graph.z*0.8 % 1 == 0, 'z*0.8 is not an integer!'
        self.inh = int(graph.z*0.8) if inh else 0
        self.t_end = t_end
        self.tails_mod = []
        self.heads_mod = []
        self.shift = shift
        self.edge_count_matrices = []
        self.z_ = np.copy(graph.z)
        self.z_old = 0
        self.z_change = 0
        self.time = 0
        self.time_axis = np.linspace(0, self.t_end, int(self.t_end/self.stepwidth))
        if (self.t_end / self.stepwidth) % 1 != 0:
            print('Attention: The time span {0} divided by the time step {1} is not an integer.'.format(
                self.t_end, self.stepwidth))
        self.tails_all = []
        self.heads_all = []
        self.resources = []

    def evaluate_perturbation(self, beta_list, graph):
        sum_beta = np.sum(beta_list)

        if graph.epsilon != 0 and sum_beta != 0:

            rate = (graph.beta + graph.epsilon)

            p = self.rate_to_prob(rate)
            graph.edges += np.where(beta_list * graph.edges == 1,
                                    -np.random.binomial(1, p, size=graph.z) * 3, 0)
            graph.edges += np.where(beta_list * graph.edges == 2,
                                    -np.random.binomial(1, 2*p-p**2, size=graph.z) * 3, 0)

            beta_list = np.where(beta_list > 0, 1, 0)
        """If there is no perturbation edges are randomly broken with probability beta"""
        if graph.beta != 0:

            graph.edges += np.where((1 - beta_list) * graph.edges == 1,
                                    -np.random.binomial(
                                        1, self.rate_to_prob(graph.beta),
                                        size=graph.z) * 3, 0)

        """During the perturbation step there is no rewiring"""
        if graph.alpha != 0 and sum_beta == 0:

            graph.edges = np.where(graph.edges == 0,
                                   np.random.binomial(
                                       1, self.rate_to_prob(graph.alpha),
                                       size=graph.z) * 5, graph.edges)
            if self.inh != 0:
                """excitatory neurons"""
                zero_indices = np.where(graph.edges[:self.inh]
                                        == 5)[0].astype(int)
                p = np.random.permutation(len(zero_indices))

                graph.heads[:self.inh][zero_indices] =\
                    graph.heads[:self.inh][zero_indices[p]]

                """inhibitory neurons"""
                zero_indices = np.where(graph.edges[self.inh:]
                                        == 5)[0].astype(int)
                p = np.random.permutation(len(zero_indices))
                graph.heads[self.inh:][zero_indices] =\
                    graph.heads[self.inh:][zero_indices[p]]
            else:
                zero_indices = np.where(graph.edges == 5)[0].astype(int)
                p = np.random.permutation(len(zero_indices))

                graph.heads[zero_indices] = graph.heads[zero_indices[p]]

            graph.edges = np.where(graph.edges == 5, 1, graph.edges)

        """The entries -2 are turned to zeros"""

        if sum_beta != 0 or graph.beta != 0:
            graph.edges = np.where(graph.edges == -2, 0, graph.edges)  # [0]
            self.resources.append(graph.z - np.sum(graph.edges))
        indices = [1 == graph.edges]

        """If this list step takes too much time, save self.tails and self.edges,
        instead and calculate the mod versions only when matrix is created"""

        self.tails_mod = graph.tails[tuple(indices)]

        self.heads_mod = graph.heads[tuple(indices)]

        self.tails_all.append(self.tails_mod)
        self.heads_all.append(self.heads_mod)

        self.time += 1

        self.z_ = graph.z - np.sum(graph.edges)

    def rate_to_prob(self, rate):
        return 1 - np.exp(-rate * self.stepwidth)
