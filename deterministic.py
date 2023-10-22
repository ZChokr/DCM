import numpy as np


class Deterministic:
    def __init__(self, graph, stepwidth, t_end, shift=False):
        """
        :param graph: graph class object
        :param stepwidth: stepwidth for Euler integration
        :param t_end: end of simulation
        :param shift: shift adjacency matrix by initial matrix zNN^T
        """

        'This code describes the simulation of the deterministic kinetic equation using the Euler method.'

        self.stepwidth = stepwidth

        self.z_change = 0.00005
        self.t_end = t_end
        self.edge_count_matrices = []

        if graph.connected == 0:
            self.initial_graph = (np.zeros([graph.n, graph.n]))
        else:
            self.initial_graph = graph.z * np.outer(graph.uniform, graph.uniform)
        self.initial_matrix = np.copy(self.initial_graph)

        self.edge_count_matrices.append(np.zeros([graph.n, graph.n]))

        self.time = 0
        self.time_axis = np.linspace(0, self.t_end, int(self.t_end/self.stepwidth))
        if (self.t_end / self.stepwidth) % 1 != 0:
            print('Attention: The time span {0} divided by the time step {1} is not an integer.'.format(self.t_end,
                                                                                                        self.stepwidth))
        self.shift = shift
        self.z_ = np.copy(graph.z)

    def euler(self, beta_matrix, graph):
        beta_matrix += graph.beta * np.ones([graph.n, graph.n])
        ddw = self.kinetic_equation(self.initial_graph, beta_matrix, graph)

        dw = self.stepwidth * ddw

        self.edge_count_matrices.append(
            self.initial_graph + dw - np.outer(graph.uniform, graph.uniform)
            * graph.z if (self.shift is True) else self.initial_graph + dw)

        self.initial_graph += dw
        self.time += 1

    def kinetic_equation(self, w, beta, graph):
        y_i_ = graph.y - np.sum(w, axis=1)

        x_j_ = graph.x - np.sum(w.T, axis=1)

        self.z_ = graph.z - np.sum(w)

        perturb = beta * w

        dw_ij_dt = graph.alpha * np.outer(y_i_, x_j_) / self.z_ - perturb\
            if self.z_ > self.z_change else -perturb

        return dw_ij_dt
