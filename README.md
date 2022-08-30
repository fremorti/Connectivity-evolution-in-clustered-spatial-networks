# Connectivity-evolution-in-clustered-spatial-networks
This repository include scripts that were used in generating and analyzing data from the manuscript titled: 'Connectivity evolution in clustered spatial networks'.
The repository include 3 pyhton scripts
* networks.py: generates networks as distance matrices from clustered spatial patterns or from a non-spatially explicit algorithm (not used in this model). This is called by the other two scripts but includes a chunck of code for generating network visualisations used in fig. S1-2, hashed out to enable functioning of the other two scripts
* landscape contours.py: generate network metric plots for distance- and cost-based networks for a range of network generating parameters, fig 2-3
* metapop.py: includes code for the IBM, data collection and visualization for a range of network generating parameters.
