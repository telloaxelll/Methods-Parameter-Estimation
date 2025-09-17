# Import needded modules 
import numpy as np
import matplotlib.pyplot as plt
from vehicles import scenario_1_data, scenario_2_data, scenario_3_data, scenario_4_data
import os

# Directory for simulation plots
plot_dir = os.path.join(os.path.dirname(__file__), "plots")

np.random.seed(41) # seed for reproducibility

time = 900 # number of time steps
t_axis = np.arange(time)

dt = 1e-2 # time step difference differential  

s_0 = 37.8 # initial space gap (60 meters)
u_0 = 33.0 # initial lead velocity (33 m/s)
v_0 = 32.5 # initial following velocity (31 m/s)

true_theta = np.array([0.08, 0.12, 1.50]) # true theta parameter from paper 

dv_max = 3.0 # maximum acceleration/deceleration (3 m/s^2)

if __name__ == "__main__":
    
    # Scenario 1: 
    # - We simulate two vehicles, where u(t) is the lead vehicle and v(t) is the ego
    #   vehicle. We model these two cars, with u(t) in the lead on a highway. We will
    #   test two cases.
    # - Case 1 | Equilibrium: Vehicles are under constant predetermined speed 
    # - Case 2 | Nonequilibrium: Vehicles that are under nonconstant determined speed 
    #            (with added fluxuations in velocities)
    print("----------Case 1: Random Walk----------")
    scenario_1_data("NON-EQ", u_0, v_0, s_0, time, dv_max, dt, true_theta)
    scenario_1_data("EQ", u_0, v_0, s_0, time, dv_max, dt, true_theta)

    # Scenario 2: 
    # - We simulate two vehicles, where u(t) is the lead vehicle and v(t) is the ego
    #   vehicle. We model these two cars, with u(t) in the lead on a highway. We will 
    #   only test one case. 
    # - Case 1 | Induced Road Curvature: Vehicles are under constant predetermined speed
    print("----------Case 2: Induced Road Curvature----------")
    scenario_2_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)

    # Scenario 3:
    # - We simulate two vehicles, where u(t) is the lead vehicle and v(t) is the ego
    #   vehicle. We model these two cars, with u(t) in the lead on a suburban road. We will
    #   test two cases.
    # - Case 1 | Suburban Environment: Vehicles are under constant predetermined speed with 
    #            more frequent stops and starts
    print("----------Case 3: Suburban Environment----------")
    scenario_3_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)
    
    # Scenario 4: 
    # - We simulate two vehicles, where u(t) is the lead vehicle and v(t) is the ego
    #   vehicle. We model these two cars, with u(t) in the lead on a highway. We will 
    #   only test one case. 
    # - Case 1 | Aggressive Behavior: Vehicles are under constant predetermined speed with 
    #            road hazards that require hard breaking. 
    print("----------Case 4: Aggressive Behavior----------")
    scenario_4_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)
