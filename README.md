# Methods for Parameter Estimation in Adaptive Cruise Control 

## Introduction

## Notebook Functionality
This repository is structured to reproduce the results presented here. Some dependencies may be outdated at the time of cloning or modification; users are advised to rely on stable, compatible package versions when running the simulations.

This section presents a general case on how things work under the hood for our simulation. Please contact [Axel Muñiz Tello](mailto:amuniztello@ucmerced.edu) for any changes to the implementation of this code. 

### Driving Scenarios
We design and implement 4 realistic scenarios that are commonly experienced when driving on a road. Further details on scenario formulations or experimental results can be found here [Robustness Analysis of Least Squares-based Adaptive Cruise Control in Real-World Scenarios](https://doi.org/10.5070/M418165653). 

In our simulations we implement:
1. Random Walk (Nonequilibrium) - nonzero changes in velocity 
2. Random Walk (Equilibrium) - velocity in the lead and following car are 0. 
3. Induced Road Curvature - induced road curvature that tests responsiveness and numerical stability of algorithms 
4. Suburban Environemnt - tests algorithms under an suburban based environment, it includes: speed bumps, stop signs, pedestrians, etc. 
5. Aggressive Lead Driver - simulates an aggressive, and erratic driver and models how robust these algorithms are under aggressive-like driving behavior. 

## References
This repository references the following research papers to conduct and implement all simulations
- Wang et al. [Online Parameter Estimation Methods for AdaptiveCruise Control Systems](lab-work.github.io/download/wang2020online.pdf)
- Muñiz Tello, Axel and Ayush Pandey[Robustness Analysis of Least Squares-based Adaptive Cruise Control in Real-World Scenarios](https://doi.org/10.5070/M418165653)