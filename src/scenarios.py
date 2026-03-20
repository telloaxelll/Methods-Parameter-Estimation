# Import Needed Dependencies: 
import numpy as np

# Case 1: Random Walk Scenario - Non-Equilibrium
def scenario_1_non_eq(u_0, v_0, s_0, time, dv_max, dt, true_theta):
    """
    Generate data for scenario 1 non-equilibrium case.
    Returns: tuple (u_t, v_t, s_t) containing the time series data
    """
    # Allocate Arrays for u_t, v_t, and s_t:
    u_t = np.zeros(time)
    v_t = np.zeros(time)
    s_t = np.zeros(time)

    u_t[0] = u_0
    v_t[0] = v_0
    s_t[0] = s_0

    for i in range(1, time):
        # Velocity Sample Generation for u(t)
        u_t[i] = u_t[i - 1] + np.random.normal(loc=0, scale=0.5)  # mean 0, std 0.5

        # Compute Current Gap and Velocity: 
        s_prev = s_t[i - 1]
        v_prev = v_t[i - 1]
        u_prev = u_t[i - 1]

        # CTH-RV Update (Following Vehicle) - v(t):
        acc = true_theta[0] * (s_prev - true_theta[2] * v_prev) + true_theta[1] * (u_prev - v_prev) 
        acc = np.clip(acc, -dv_max, dv_max)  # apply physical constraint

        # Update Follower Velocity and Space Gap
        v_t[i] = v_prev + acc * dt
        s_t[i] = s_prev + (u_prev - v_prev) * dt 

    return u_t, v_t, s_t

# Case 1: Random Walk Scenario - Equilibrium
def scenario_1_eq(u_0, v_0, s_0, time, dv_max, dt, true_theta):
    """
    Generate data for scenario 1 equilibrium case.
    Returns: tuple (u_t, v_t, s_t) containing the time series data
    """
    # Allocate Arrays for u_t, v_t, and s_t:
    u_t = np.zeros(time)
    v_t = np.zeros(time)
    s_t = np.zeros(time)

    # Set initial conditions for equilibrium
    equilibrium_speed = 30.0
    u_t[0] = equilibrium_speed
    v_t[0] = equilibrium_speed
    s_t[0] = true_theta[2] * equilibrium_speed  # s = τv

    # Generate equilibrium samples
    for i in range(1, time):
        u_t[i] = u_t[i - 1] + np.random.normal(0, 0.05)  # small noise for realism

        s_prev = s_t[i - 1]
        v_prev = v_t[i - 1]
        u_prev = u_t[i - 1]

        acc = true_theta[0] * (s_prev - true_theta[2] * v_prev) + true_theta[1] * (u_prev - v_prev)
        acc = np.clip(acc, -dv_max, dv_max)

        v_t[i] = v_prev + acc * dt
        s_t[i] = s_prev + (u_prev - v_prev) * dt

    return u_t, v_t, s_t

# Case 2: Induced Curved Road
def scenario_2_data(u_0, v_0, s_0, time, dv_max, dt, true_theta):
    """
    Generate data for scenario 2: curved road with Gaussian speed profile.
    
    Case Details: 
    - Non-equilibrium case simulation since equilibrium (v_k = u_k) would create
      a rank deficient matrix where rank(X) = 1 << 3, leading to infinite solutions
      and ill-posed estimation problem.
    
    Returns: tuple (u_t, v_t, s_t) containing the time series data
    """
    u_t = np.zeros(time)
    v_t = np.zeros(time)
    s_t = np.zeros(time)

    u_t[0] = u_0
    v_t[0] = v_0
    s_t[0] = s_0

    center = time // 2  # midpoint of time
    curve_width = 100   # controls tightness of curve
    min_speed = 20      # slowest point on the curve (m/s)

    for i in range(1, time):
        # u_t will be modeled after the "Gaussian dip"
        u_t[i] = u_0 - (u_0 - min_speed) * np.exp(-((i - center) ** 2) / (2 * curve_width ** 2))

        s_prev = s_t[i - 1]
        v_prev = v_t[i - 1]
        u_prev = u_t[i - 1]

        # Compute acceleration with CTH-RV model
        acc = true_theta[0] * (s_prev - true_theta[2] * v_prev) + true_theta[1] * (u_prev - v_prev)
        acc = np.clip(acc, -dv_max, dv_max)

        v_t[i] = v_prev + acc * dt
        s_t[i] = s_prev + (u_prev - v_prev) * dt

    return u_t, v_t, s_t

# Case 3: Suburban Environment 
def scenario_3_data(u_0, v_0, s_0, time, dv_max, dt, true_theta):
    """
    Generate data for scenario 3: suburban environment with stop signs, pedestrian zones, and stoplights.
    Returns: tuple (u_t, v_t, s_t) containing the time series data
    """
    u_t = np.zeros(time)
    v_t = np.zeros(time)
    s_t = np.zeros(time)

    u_t[0] = u_0
    v_t[0] = v_0
    s_t[0] = s_0

    cruise_speed = 13.0
    noise_std = 0.2
    decel_rate = -3.0
    accel_rate = 1.5
    stop_duration = 3

    np.random.seed(42)

    stop_signs = np.random.choice(range(100, time - 100), size=3, replace=False)
    pedestrian_zones = np.random.choice(range(100, time - 100), size=2, replace=False)
    stoplights = np.random.choice(range(100, time - 100), size=2, replace=False)

    for i in range(1, time):
        u_t[i] = u_t[i-1] + np.random.normal(0, noise_std)

        # Hard Stop for Stop Signs
        if i in stop_signs:
            for k in range(stop_duration):
                if i + k < time:
                    u_t[i + k] = max(0, u_t[i + k - 1] + decel_rate * dt)

        # Pedestrian Slow Zone:
        if i in pedestrian_zones:
            u_t[i] = u_t[i-1] - 1.0

        # Stochastic Stoplight:
        if i in stoplights and np.random.rand() < 0.5:  # 50% chance of red light
            red_duration = np.random.randint(3, 8)
            for k in range(red_duration):
                if i + k < time:
                    u_t[i + k] = 0.0

        u_t[i] = np.clip(u_t[i], 0, 20)

        # Follower car dynamics
        s_prev = s_t[i - 1]
        v_prev = v_t[i - 1]
        u_prev = u_t[i - 1]

        acc = true_theta[0] * (s_prev - true_theta[2] * v_prev) + true_theta[1] * (u_prev - v_prev)
        acc = np.clip(acc, -dv_max, dv_max)

        v_t[i] = v_prev + acc * dt
        
        # *** FIX 1: Prevent follower from exceeding lead vehicle ***
        v_t[i] = min(v_t[i], u_t[i])
        
        # *** FIX 2: Ensure follower velocity is non-negative ***
        v_t[i] = max(v_t[i], 0.0)
        
        s_t[i] = s_prev + (u_prev - v_prev) * dt
        
        # *** FIX 3: Prevent negative space gap (collision) ***
        if s_t[i] < 0:
            s_t[i] = 0.0
            v_t[i] = u_t[i]  # Match lead vehicle speed to maintain zero gap

    return u_t, v_t, s_t

# Case 4: Aggressive Lead Driver
def scenario_4_data(u_0, v_0, s_0, time, dv_max, dt, true_theta):
    """
    Generate data for scenario 4: aggressive lead driver with sudden acceleration and braking events.
    Returns: tuple (u_t, v_t, s_t) containing the time series data
    """
    u_t = np.zeros(time)
    v_t = np.zeros(time)
    s_t = np.zeros(time)

    u_t[0] = u_0
    v_t[0] = v_0
    s_t[0] = s_0

    np.random.seed(0)

    cruise_speed = 33.0
    noise_std = 0.3
    accel_spikes = np.random.choice(range(100, time - 100), size=5, replace=False)
    brake_spikes = np.random.choice(range(100, time - 100), size=5, replace=False)

    for i in range(1, time):
        # Default: noisy cruise
        u_t[i] = u_t[i-1] + np.random.normal(0, noise_std)

        # Sudden acceleration event
        if i in accel_spikes:
            for k in range(5):  # short burst
                if i + k < time:
                    u_t[i + k] = min(u_t[i + k - 1] + 4.0 * dt, 40.0)

        # Sudden braking event
        if i in brake_spikes:
            for k in range(5):
                if i + k < time:
                    u_t[i + k] = max(u_t[i + k - 1] - 5.0 * dt, 0.0)

        # Clamp speed
        u_t[i] = np.clip(u_t[i], 0, 40)

        # Follower dynamics
        s_prev = s_t[i - 1]
        v_prev = v_t[i - 1]
        u_prev = u_t[i - 1]

        acc = true_theta[0] * (s_prev - true_theta[2] * v_prev) + true_theta[1] * (u_prev - v_prev)
        acc = np.clip(acc, -dv_max, dv_max)

        v_t[i] = v_prev + acc * dt
        s_t[i] = s_prev + (u_prev - v_prev) * dt

    return u_t, v_t, s_t