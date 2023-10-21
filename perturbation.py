import numpy as np
import itertools as it


class Perturbation:
    def __init__(self, graph, numPerturbations,
                 rates=None, perturbationMode=None,
                 perturbationTime=None, MC=True, perturbationType='vertices'):
        """
        :param graph: Graph object
        :param numPerturbations: number of perturbations
        :param rates: dictionary of rates: rewiring rate alpha,
                    random breaking rate beta
                    and rate of explicit perturbation
                    epsilon = number of edges to be broken
                    in det case
        :param perturbationMode:
                        {'mode': 'randomPerturbation', 'arg(percentage)': } or
                        {'mode': 'explicitPerturbation', 'arg(vertices/pairs)': } or
                              None = no perturbation,
                              or {'mode': '80_20_Perturbation', 'arg(vertices(pairs)':}
        :param perturbationTime: {'manuel': list of moments of perturbations},
                    {'automatic': True/False, next perturbation starts when
                     equilibrium found}
        :param MC: True or False
        :param perturbationType:   'vertex' ^= arg(vertices) or 'feedforward' ^= arg(list of index pairs)
        """
        self.numPerturbations = numPerturbations
        self.alpha = rates['alpha']
        self.beta = rates['beta']
        self.epsilon = rates['epsilon']
        self.perturbationMode = perturbationMode
        self.perturbationTime = perturbationTime
        self.perturbationType = perturbationType
        self.P_vectors = []
        self.P_row_vectors = []
        self.P_col_vectors = []
        self.perturbation_matrix = []
        self.perturbedVertices = []
        self.perturbedList = []
        self.MC = MC
        if self.numPerturbations != 0:
            for numperturb in range(self.numPerturbations):
                if self.perturbationMode['mode'] == 'randomPerturbation':
                    self.perturbedVertices.append(
                        np.random.choice(range(graph.n),
                                         graph.n*self.perturbationMode['arg']))
                elif self.perturbationMode['mode'] == 'explicitPerturbation':
                    self.perturbedVertices.append(self.perturbationMode['arg'][numperturb])
                elif self.perturbationMode['mode'] == '80_20_Perturbation':
                    perVer = np.array(self.perturbationMode['arg'][numperturb])
                    print('here', len(perVer), len(perVer) % 5 == 0)
                    assert len(perVer) % 5 == 0, 'number of perturbed vertices cannot' \
                                                 ' be distributed correctly.'

                    perVer[int(len(perVer) * 0.8):] += int(graph.n * 0.8) - \
                                                       perVer[int(len(perVer) * 0.8)]\
                                                       + int(perVer[0] * 0.2)
                    self.perturbedVertices.append(perVer)
                else:
                    print("This perturbation mode, does not exist.")

                if self.perturbationType == 'vertices' and self.numPerturbations != 0:
                    P = np.zeros(graph.n)
                    PP = np.zeros(graph.n)
                    P[self.perturbedVertices[numperturb]] =\
                        1/np.sum(len(self.perturbedVertices[numperturb]))

                    PP[self.perturbedVertices[numperturb]] = 1
                    self.P_vectors.append(P)
                    self.perturbation_matrix.append(1/2 * (np.outer(np.ones(graph.n), PP) +
                                                    np.outer(PP, np.ones(graph.n))))
                elif self.perturbationType == 'feedforward' and self.numPerturbations != 0:
                    P_row = np.zeros(graph.n)
                    PP_row = np.zeros(graph.n)
                    P_col = np.zeros(graph.n)
                    PP_col = np.zeros(graph.n)
                    P_row[self.perturbedVertices[numperturb][0]] = \
                        1 / np.sum(len(self.perturbedVertices[numperturb][0]))
                    P_col[self.perturbedVertices[numperturb][1]] = \
                        1 / np.sum(len(self.perturbedVertices[numperturb][1]))
                    PP_row[self.perturbedVertices[numperturb][0]] = 1
                    PP_col[self.perturbedVertices[numperturb][1]] = 1
                    self.P_row_vectors.append(P_row)
                    self.P_col_vectors.append(P_col)
                    self.perturbation_matrix.append(1 / 2 * (np.outer(np.ones(graph.n), PP_col) +
                                                             np.outer(PP_row, np.ones(graph.n))))

            self.P_vectors = np.array(self.P_vectors)
            self.P_row_vectors = np.array(self.P_row_vectors)
            self.P_col_vectors = np.array(self.P_col_vectors)
            self.perturbation_matrix = np.array(self.perturbation_matrix)
            self.beta_matrix = self.epsilon*self.perturbation_matrix
            if self.beta != 0:
                self.beta_matrix += self.beta * np.ones([graph.n, graph.n])

    def perturb_MC(self, graph, numperturb):
        newlist = np.zeros([graph.z])

        if self.perturbationType == 'vertices':

            for i in self.perturbedVertices[numperturb]:
                newlist += np.where(graph.heads == i, 1, 0) + \
                           np.where(graph.tails == i, 1, 0)

        elif self.perturbationType == 'feedforward':
            for i, j in it.zip_longest(
                    self.perturbedVertices[numperturb][0],
                    self.perturbedVertices[numperturb][1]):
                newlist += np.where(graph.heads == i, 1, 0) + \
                           np.where(graph.tails == j, 1, 0)

        perturbedList = newlist

        return perturbedList
