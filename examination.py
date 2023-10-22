import numpy as np
import scipy.linalg as sp


def zeroth_col_row(graph, matrix, col_sums, row_sums):
    a = matrix
    b = graph.z / graph.n - col_sums
    c = graph.z / graph.n - row_sums
    last_entry = graph.z / graph.n - np.sum(c)
    c = np.append(c, last_entry)
    d = np.vstack((a, b))
    e = np.vstack([d.T, c]).T
    return e


class Examination:
    def __init__(self, graph, time_evolution, indices,
                 inhibition=0,  eigenvalues=True, perturbation=None,
                 trace=True, coefficient_matrix=False, zeroth_row_col=False,
                 gain=0.00125):
        """
        :param graph: Graph object
        :param time_evolution: class object
        :param indices: points in time for which matrices should be calculated to be examined
        :param eigenvalues: whether to calculate the eigenvalues or not
        :param inhibition: weight multiplied to the inhibitory edges, positive number must
                be included to get negative weight
        :param perturbation: class object
        :param trace: whether to calculate the trace or not
        :param coefficient_matrix: whether to multiply the gain to the edge count matrix and then subtract
                a unit matrix from it
        :param zeroth_row_col: whether to include a zeroth row to the edge count matrix that contains the number
                of free heads or tails
        :param gain: the constant multiplied to the edge count matrix when creating the coefficient matrix
        """

        """Functions to examine the edge count matrix and its eigenvalue spectrum"""

        self.gain = gain
        self.coefficient_mat = coefficient_matrix
        self.graph = graph
        self.perturbed_edges = perturbation.perturbationMode['arg'] if perturbation is not None else None
        self.perturbed_time = perturbation.perturbationTime['manuel'] if perturbation is not None else None

        if graph.MC:
            self.type = time_evolution.type
            self.type_name = time_evolution.type_name
        else:
            self.type = time_evolution.type
            self.type_name = time_evolution.type_name

        self.inhibition = inhibition
        self.zeroth_row_col = zeroth_row_col
        self.time_evolution = time_evolution
        self.eigenvalues = []
        self.eigenvectors = []
        self.traces = []
        self.t = self.type.time_axis
        self.connected_edges = (self.calculate_connected_edges())
        self.free_edges = graph.z - self.connected_edges
        self.edge_count_matrix = np.zeros([graph.n, graph.n])
        self.edge_count_matrices = []
        self.indices = indices
        self.trace = trace

        if self.type_name == 'MonteCarlo':
            self.type.edge_count_matrices = []
            for ind in self.indices:
                self.create_edge_count_matrix(graph, int(ind/self.type.stepwidth))
                self.edge_count_matrices = np.copy(self.type.edge_count_matrices)
        else:
            ind = (np.array(self.indices)/self.type.stepwidth).astype(int)
            self.edge_count_matrices =\
                np.copy(np.array(self.type.edge_count_matrices)[ind])

        if self.inhibition != 0:
            self.inh = int(graph.n * 0.8)
            self.edge_count_matrices = self.inhibition_func(self.edge_count_matrices)

        if eigenvalues:
            for num, ind in enumerate(indices):

                ev, vec = self.calculate_eigenvalues(
                        self.edge_count_matrices[num])
                self.eigenvalues.append(ev)
                self.eigenvectors.append(vec)

        if trace:
            self.traces = self.calculate_trace()

    def calculate_eigenvalues(self, matrix):
        eigenvalues, eigenvectors = sp.eig(self.coefficient_matrix(matrix), left=True, right=False)
        return np.array(eigenvalues), np.array(eigenvectors)

    def coefficient_matrix(self, matrix):
        if self.coefficient_mat:
            print('coefficient matrix on')
            return self.gain * matrix - np.identity(self.graph.n)
        else:
            return matrix

    def inhibition_func(self, matrix):
        """Turn the last 80 percent of neurons to inhibitory neurons by multiplying -weight
         to the corresponding matrix columns"""
        for ind in range(len(matrix)):
            matrix[ind][:, self.inh:] = -self.inhibition * matrix[ind][:, self.inh:]
        return matrix

    def create_edge_count_matrix(self, graph, i):
        self.edge_count_matrices = []
        self.edge_count_matrix = np.zeros([graph.n, graph.n])
        """Create adjacency matrix"""
        np.add.at(self.edge_count_matrix, (self.type.heads_all[i],
                                           self.type.tails_all[i]), 1)
        ad_sum = np.sum(self.edge_count_matrix)

        assert ad_sum <= graph.z, \
            "Sum of entries should be equal or" \
            " less than {0} but is {1}".format(graph.z, ad_sum)
        col_sums = np.sum(self.edge_count_matrix, axis=0)
        row_sums = np.sum(self.edge_count_matrix, axis=1)

        if self.connected_edges[i] == graph.z:
            assert np.all(col_sums == graph.z/graph.n),\
                "Column sums are not equal to z/n"
            assert np.all(row_sums == graph.z/graph.n), \
                "Row sums are not equal to z/n"

        if self.type.shift:
            self.edge_count_matrix = self.edge_count_matrix - \
                                     np.outer(graph.uniform, graph.uniform) * graph.z

        if self.zeroth_row_col:
            self.edge_count_matrix = zeroth_col_row(graph, self.edge_count_matrix, col_sums, row_sums)

        self.type.edge_count_matrices.append(self.edge_count_matrix)

    def calculate_connected_edges(self):
        counts = 0
        if self.type_name == 'MonteCarlo':

            counts = np.array([len(x) for x in self.type.tails_all])

        elif self.type_name == 'deterministic':
            counts = np.sum(np.sum(self.type.edge_count_matrices, axis=1), axis=1)
        return counts

    def examine_connectivity(self):
        """Examination of the connectivity of the different blocks in the edge count matrix"""
        if self.perturbed_edges is not None:
            Wii_list = []
            Wij_list = []
            Wjj_list = []
            count = 0
            indices = self.perturbed_edges[count]
            Wii_0 = np.sum(self.edge_count_matrices[0][indices, indices])
            wij_0 = (np.sum(self.edge_count_matrices[0][indices])
                     + np.sum(self.edge_count_matrices[0].T[indices]))
            Wij_0 = wij_0 - 2 * Wii_0
            Wjj_0 = np.sum(self.edge_count_matrices[0]) - Wij_0 - Wii_0

            for i in self.indices:
                Wii = np.sum(self.edge_count_matrices[i][indices, indices])
                wj = self.edge_count_matrices[i][indices]
                wi = self.edge_count_matrices[i].T[indices]
                wij = np.sum(wi) + np.sum(wj)
                Wij = wij - 2 * Wii
                Wjj = np.sum(self.edge_count_matrices[i]) - Wij - Wii
                resources = self.type.resources[count]

                Wii_list.append((Wii - Wii_0) / resources)
                Wij_list.append((Wij - Wij_0) / resources)

                Wjj_list.append((Wjj - Wjj_0) / resources)

                if i == self.perturbed_time[count+1]:
                    count += 1
            return Wii_list, Wij_list, Wjj_list
        else:
            print('Connectivity cannot be examined if there is no perturbation.')

    def calculate_trace(self):
        if self.type_name == 'deterministic':

            return np.trace(self.coefficient_matrix(
                np.copy(self.time_evolution.ad_list_or_matrix)), axis1=1, axis2=2)
        else:
            """Calculate the trace without creating the matrix"""
            self.traces = []
            for tails, heads in zip(self.time_evolution.ad_list_or_matrix1,
                                    self.time_evolution.ad_list_or_matrix2):

                a = np.array(heads) - np.array(tails)

                if self.inhibition != 0:
                    inh = np.argmax(heads > int(self.graph.n*0.8)-1)

                    u = np.count_nonzero(a[:inh] == 0)
                    uu = -self.inhibition * np.count_nonzero(a[inh:] == 0)
                    print(u, uu, self.inhibition)

                else:
                    u = np.count_nonzero(a == 0)
                    uu = 0
                if self.coefficient_mat:
                    self.traces.append(self.gain*(u+uu)-1)
                else:
                    self.traces.append(u + uu)

            return self.traces
