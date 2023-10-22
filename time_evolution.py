import numpy as np


class TimeEvolution:
    def __init__(self, perturbation, graph, types, reset=False, MC=True):
        self.perturbation_time = []
        if MC:
            self.type_name = 'MonteCarlo'
            self.type = types
        else:
            self.type_name = 'deterministic'
            self.type = types
        self.t = self.type.time_axis
        self.stepwidth = self.type.stepwidth
        self.reset = reset

        if self.type_name == 'deterministic':
            self.function = self.type.euler
            self.perturb_list_or_matrix = perturbation.beta_matrix \
                if perturbation is not None else np.zeros([graph.n, graph.n])
            self.ad_list_or_matrix = self.type.edge_count_matrices

        elif self.type_name == 'MonteCarlo':
            self.function = self.type.evaluate_perturbation
            self.perturb_list_or_matrix = perturbation.perturbedList\
                if perturbation is not None else np.zeros([graph.z])

            self.ad_list_or_matrix1 = self.type.tails_all
            self.ad_list_or_matrix2 = self.type.heads_all

        """
        In order to start the simulation at a state of equilibrium,
         a pre-simulation is run,
        until all edges are connected
        """

        if graph.connected != 0:
            for repeat in range(1):
                self.function(0, graph)

        """
        The first adjacency matrix at time step 0 is set to the last matrix obtained in
        the pre-simulation
        """

        if self.type_name == 'deterministic':

            last_entry1 = np.copy(self.ad_list_or_matrix[self.type.time - 1])
            if graph.connected != 0:
                last_entry1 = self.type.initial_graph
            self.ad_list_or_matrix = self.type.edge_count_matrices = []
            self.ad_list_or_matrix.append(self.type.initial_matrix)

        elif self.type_name == 'MonteCarlo':
            """
            Since the entries of self.heads_all and self.tails_all have different sizes,
            one can not use simple entry assignment, therefore append is used instead.
            """

            if graph.connected != 0:
                last_entry1 = np.copy(self.ad_list_or_matrix1[self.type.time - 1])
                last_entry2 = np.copy(self.ad_list_or_matrix2[self.type.time - 1])

                self.ad_list_or_matrix1 = self.type.tails_all = []
                self.ad_list_or_matrix2 = self.type.heads_all = []
                self.ad_list_or_matrix1.append(last_entry1)
                self.ad_list_or_matrix2.append(last_entry2)
            else:
                self.ad_list_or_matrix1 = self.type.tails_all = []
                self.ad_list_or_matrix2 = self.type.heads_all = []
                self.ad_list_or_matrix1.append([])
                self.ad_list_or_matrix2.append([])
            """
            It is possible to write it this way since presim has only
             one step due to the
            fact that the initial self.edge vector
             has ones everywhere -> graph already
            completely connected.
            """

        """
        If the perturbations should be such that they start whenever
        the equilibrium state is
        reobtained one sets the 'automatic' option to True
        """

        """code removed: please contact us if you are interested"""

        """
        If one wants to set the perturbation times manually, 'automatic'
         is set to False.
        """
        perturbnum = 0

        for t in range(int(self.type.t_end / self.type.stepwidth) - 1):
            """
            If the current timestep equals the preset moments of perturbation
            a perturbation occurs, else, solely rewiring occurs.
            When there is a perturbation, there occurs always rewiring at the 
            same time. This can be changed if not wanted.
            """

            if perturbation is not None and (perturbation.perturbationTime['manuel'][perturbnum][0]
                    <= self.type.time * self.type.stepwidth <=
                    perturbation.perturbationTime['manuel'][perturbnum][-1]):

                if self.type_name == 'MonteCarlo':
                    self.perturb_list_or_matrix.append(
                        perturbation.perturb_MC(graph, perturbnum))
                self.function(self.perturb_list_or_matrix[perturbnum], graph)

                self.perturbation_time.append(self.type.time*self.type.stepwidth)

            else:
                self.function(0, graph)

            """
            If the end of one perturbation cycle is reached, the next
             one is started.
            """
            if perturbation is not None and perturbnum < perturbation.numPerturbations - 1 \
                    and (t * self.type.stepwidth) \
                    == perturbation.perturbationTime['manuel'][perturbnum][-1]:
                perturbnum += 1
