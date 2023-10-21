# DCM
An algorithm for a Monte Carlo simulation describing the self organization of a network

## Description
This code presents an implementation of a dynamic directed configuration model.
The basic algorithm works as follow:
- A number of vertices n and a number of edges z is chosen
- A number of in - and outdegrees for each vertex is chosen such that the sum of degrees is given by z, respectively.
- Lists of length z containing all indegrees and outdegrees of each vertex are created.
- Edges can be created and reconnected. These processes are led by a breaking and a rewiring rate.
- In each time step of the simulation a percentage of edges is broken and/or rewired. This percentage is given by the corresponding rate.
- Instantaneous perturbations can be introduced by choosing a point in time.
  
Additionally to the Monte Carlo simulation, the solutions of the deterministic equation are simulated using the Euler method.

A tutorial file that shows how the code works is included.

## Authors
Zainab Chokr 

## Sources
The considerations follow the paper "Associative remodeling and repair in self-organizing neuronal networks (2018)" by Nebojša Gašparović, Júlia V. Gallinaro and Stefan Rotter.

## Acknowledgments
This work was supervised by Prof. Dr. Rotter from the Bernstein Center in Freiburg.
