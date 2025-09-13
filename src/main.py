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

dt = 1e-1 # time step difference differential  

s_0 = 50.0 # initial space gap (60 meters)
u_0 = 33.0 # initial lead velocity (33 m/s)
v_0 = 31.0 # initial following velocity (31 m/s)

true_theta = np.array([0.08, 0.12, 12.0]) # true theta parameter 

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
    print("----------Case 2: Induced Road Curvature----------")
    scenario_2_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)

    print("----------Case 3: Suburban Environment----------")
    scenario_3_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)
   
    print("----------Case 4: Aggressive Behavior----------")
    scenario_4_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)
