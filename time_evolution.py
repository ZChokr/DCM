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
        if perturbation is not None and perturbation.perturbationTime['automatic']:

            for perturbs in range(perturbation.numPerturbations):

                self.type.count = 0
                """
                One perturbation and rewiring process runs as long as the while loop is
                True, this is the case, when the number of free edges still changes
                remarkably or the time to which the first perturbation is set
                is not achieved yet or the counts are smaller then one which means that
                the next perturbation just started.
                """

                perturbnum = 0
                while np.abs(self.type.z_) > \
                        self.type.z_change or self.type.time <= \
                        perturbation.perturbationTime['manuel'][0][0] \
                        / self.type.stepwidth + 1 \
                        or self.type.count <= 1 and self.type.time < self.type.t_end - 3:
                    self.type.z_old = np.copy(self.type.z_)
                    """
                    If the number of counts is zero or the preset perturbation
                    time for the first perturbation is reached,
                    the matrix is perturbed, else it will
                    solely rewire.
                    """
                    if self.type.time * self.type.stepwidth == \
                            perturbation.perturbationTime['manuel'][0][0] or \
                            (self.type.count == 0 and self.type.time != 1):
                        if self.type_name == 'MonteCarlo':

                            self.perturb_list_or_matrix.append(
                                perturbation.perturb_MC(graph, perturbnum))

                        self.function(self.perturb_list_or_matrix[perturbs], graph)
                        self.perturbation_time.append(
                            self.type.time*self.type.stepwidth)
                        perturbnum += 1
                    else:
                        self.function(0, graph)
                    self.type.count += 1

            """
            When all perturbations are over and the equilibrium state
             is reached again,
            all remaining edge_count_matrices are set to the value of the
            last edge_count_matrix from the simulation.
            """

            for matrix in range(self.type.time,
                                int(self.type.t_end / self.type.stepwidth)):
                if self.type_name == 'deterministic':
                    self.ad_list_or_matrix.append(
                        self.ad_list_or_matrix[self.type.time - 1])
                elif self.type_name == 'MonteCarlo':
                    self.ad_list_or_matrix1.append(
                       self.ad_list_or_matrix1[self.type.time - 1])
                    self.ad_list_or_matrix2.append(
                       self.ad_list_or_matrix2[self.type.time - 1])
        else:
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
