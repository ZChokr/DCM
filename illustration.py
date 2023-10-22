import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


class Illustration:
    def __init__(self, perturbation, examination):
        self.examination = examination
        self.time_evolution = examination.time_evolution
        self.perturbation = perturbation
        self.graph = examination.graph

    def plot_connected_edges(self):
        plt.plot(self.examination.t, self.examination.connected_edges, 'o-',
                 markersize=3, label=self.examination.type_name +
                 ' dt = {}'.format(self.time_evolution.stepwidth))
        plt.title(r'Time evolution of the connected'
                  r' edges of the system with $\alpha = {0}$,'
                  r' $\beta = {1}$ and $\gamma = {2}$'
                  .format(self.graph.alpha, self.graph.beta, self.graph.epsilon))
        for i in self.time_evolution.perturbation_time:
            plt.axvline(i-1, linestyle='--', color='grey')
        plt.xlabel('time t in s')
        plt.ylabel('number of connected edges')
        plt.legend()

    def plot_eigenvalues(self):
        for i in range(len(self.examination.indices)):
            # plt.figure(figsize=[10, 6])
            plt.plot(self.examination.eigenvalues[i].real,
                     self.examination.eigenvalues[i].imag, '*')
        plt.axis('scaled')

    def plot_traces(self, greyline=True):
        plt.plot(self.examination.t, self.examination.traces, 'o-',
                 markersize=3, label=self.examination.type_name +
                 ' dt = {}'.format(self.time_evolution.stepwidth))

        plt.title(r'Time evolution of the trace'
                  r' of the edge count matrices of the system with $\alpha = {0}$,'
                  r' $\beta = {1}$ and $\gamma = {2}$'
                  .format(self.perturbation.alpha,
                          self.perturbation.beta, np.round(self.perturbation.epsilon, 3)))
        if greyline:
            for i in self.time_evolution.perturbation_time:
                plt.axvline(i - 1, linestyle='--', color='grey')
        plt.xlabel('time t in s')
        plt.ylabel('Traces')
        plt.legend()

    def plot_connected_edges_and_traces(self, together=True, greyline=True):
        t = self.examination.t
        if together:
            fig, ax1 = plt.subplots(figsize=[8, 4])

            color = 'tab:green'
            # plt.title('The time evolution of the trace Tr(W) and the number of connected edges')
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel(r'$z-\hat{z}$', color=color)
            ax1.plot(t, self.examination.connected_edges, 'o-', markersize=2, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            # ax1.grid(color=color)
            if greyline:
                if self.perturbation is not None and self.perturbation.numPerturbations != 0:
                    for i in self.time_evolution.perturbation_time:
                        ax1.axvline(np.round(i-self.time_evolution.stepwidth), linestyle='--', color='grey')
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Trace(W)', color=color)
            # we already handled the x-label with ax1
            ax2.plot(t, self.examination.traces, 'o-', markersize=2, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax1.ticklabel_format(useOffset=False)
            ax2.ticklabel_format(useOffset=False)
            ax1.grid()
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(1, 2, figsize=[15, 4])
            ax[0].plot(t, self.examination.connected_edges, 'o-',
                       markersize=3, label=self.examination.type_name +
                       ' dt = {}'.format(self.time_evolution.stepwidth))
            if greyline:
                if self.perturbation is not None and self.perturbation.numPerturbations != 0:
                    for i in self.time_evolution.perturbation_time:
                        ax[0].axvline(i - 1, linestyle='--', color='grey')
                        ax[1].axvline(i - 1, linestyle='--', color='grey')
            ax[1].plot(t, self.examination.traces, 'o-',
                       markersize=3, label=self.examination.type_name +
                       ' dt = {}'.format(self.time_evolution.stepwidth))
            ax[0].set_xlabel('time (s)')
            ax[0].set_ylabel('Connected edges')
            ax[1].set_xlabel('time (s)')
            ax[1].set_ylabel('Trace(W)')
            ax[0].legend()
            ax[1].legend()

    def plot_edge_count_matrix(self):

        for num, ind in enumerate(self.examination.indices):
            plt.figure(figsize=[10, 6], dpi=600)
            matrix = self.examination.edge_count_matrices[num]
            cmap = 'seismic' if np.any(matrix < 0) else 'Reds'

            v_max = np.max(np.abs(matrix))

            v_min = -v_max if np.any(matrix < 0) else 0
            plt.matshow(matrix, vmin=v_min, vmax=v_max, cmap=cmap)
            plt.title(r'Edge count matrix, $W$')
            plt.colorbar()

    def plot_matrix_and_eigenvalues(self, outlier=None, radius=False, zero_line=False):
        z = self.graph.z
        n = self.graph.n
        for num, ind in enumerate(self.examination.indices):
            fig, ax = plt.subplots(1, 2, figsize=[15, 4])
            matrix = self.examination.edge_count_matrices[num]
            ev = self.examination.eigenvalues[num]
            cmap = 'seismic' if np.any(matrix < 0) else 'Reds'

            v_max = np.max(np.abs(matrix))

            v_min = -v_max if np.any(matrix < 0) else 0
            mat = ax[0].matshow(matrix, vmin=v_min, vmax=v_max, cmap=cmap)
            ax[0].set_title(r'Edge count matrix, $W$')
            fig.colorbar(mat, ax=ax[0])

            ax[1].set_title('Eigenvalue spectrum of the adjacency matrix')
            ax[1].grid()
            if zero_line:
                ax[1].axvline(0, linestyle='--', color='black')
            if outlier == None:
                ax[1].plot(ev.real, ev.imag, '*', markersize=3)
            else:
                ax[1].plot(ev.real[1:], ev.imag[1:], '*', markersize=3)

            if radius:
                radius3 = np.sqrt((z ** 2 / (z - 1) * (
                            1 / n - 1 / n ** 2) ** 2) * n)
                circ3 = plt.Circle((0, 0), radius=radius3,
                                   lw=2, edgecolor='darkred',
                                   facecolor='None', zorder=3, linestyle='dashed')

                radius = np.sqrt(np.sum(np.var(matrix, axis=0)))
                radius2 = np.max(np.sqrt(ev[1:].real ** 2 + ev[1:].imag ** 2))

                circ = plt.Circle((0, 0), radius=radius,
                                  lw=2, edgecolor='yellow',
                                  facecolor='None', zorder=3)
                circ2 = plt.Circle((0, 0), radius=radius2,
                                   lw=2, edgecolor='gray',
                                   facecolor='None', zorder=3, linestyle='dashed')
                ax[1].add_patch(circ)
                ax[1].add_patch(circ2)
                ax[1].add_patch(circ3)
            ax[1].set_xlabel(r'$Re(\lambda)$')
            ax[1].set_ylabel(r'$Im(\lambda)$')
            ax[1].axis('scaled')

    def animate_eigenvalues(self, ev):
        graph = np.array(self.examination.eigenvalues)
        ev = np.array(ev)
        fig, ax = plt.subplots(figsize=[10, 6])
        plt.close()
        plt.rcParams['savefig.facecolor'] = 'white'

        def update(i):
            ax.clear()
            M = graph[i]
            EV = ev[i]

            ax.plot(M.real, M.imag, '*', markersize=10, alpha=0.6, color='orange')
            ax.plot(EV.real, EV.imag, '*', color='black')

            ax.axis('scaled')

        anim = animation.FuncAnimation(fig, update, frames=len(graph), interval=200)
        return anim
