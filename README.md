# smooth-deep-learning
A smooth optimization perspective on Deep (Reinforcement) Learning.  
This repository contains demonstration scripts (matlab and python) for our poster.

#### Folder "matlab":

A single script “cvpr18_fnn.m”, which shows the quadratic convergence of the proposed algorithm on a reduced four region classification problem. This script can be run directly and produces immediately a plot showing the distances of individual weights to the ground truth.

#### Folder "python":
* "gauss_newton_for_deep_learning.py"
Same demonstration as in the matlab script. Additionally, methods to train a FNN classifier on the full-sized four region problem are available. At the end of this file more detailed instructions on how to use the code are provided.

* "gauss_newton_for_deep_reinforcement_learning.py"
Shows the application of the Gauss Newton algorithm to Deep Reinforcement Learning. Contains two demonstrations:  
-- policy iteration  
-- online off policy learning  
Again at the end of this file more detailed instructions on how to use the code are provided.
