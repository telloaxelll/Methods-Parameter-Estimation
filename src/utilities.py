import numpy as np
import matplotlib.pyplot as plt

def invert_gamma(gamma, dt):
    """
    Recovers alpha, beta, and tau from gamma coefficients.   

    Args:
    -----
        gamma (array - length: 3): Gamma array estimated via recursive least squares algorithm. 
        dt (float): Used measured frequency from (Wang et al., 2020).

    Returns:
    --------
        array: Array of length 3 containing the coefficient values for alpha, beta, and tau.

    Notes: 
    ------
        Refer to `rls.ipynb` for mathematical derivation of alpha, beta, and tau from gamma vector.
    """
    gamma1, gamma2, gamma3 = gamma

    alpha = gamma2/dt
    beta  = gamma3/dt

    # Avoid dividing by very small value of gamma2
    if abs(alpha) < 1e-8: 
        tau = 0.0 # fallback in case of instability
    else:
        tau = ((1 - gamma1 - gamma3) / gamma2)
    return alpha, beta, tau


def rls_filter(u_t, v_t, s_t, N, dt, true_theta, label):
    """
    Executes the recursive least squares (RLS) filter algorithm for parameter estimation of ACC systems. 

    Args:
    -----
        u_t (array - length: 900): Array containing velocity values at time (t) of the leading vehicle. 
        v_t (array - length: 900): Array cotnaining velocity values at time (t) of the following (ego) vehicle. 
        s_t (array - length: 900): Array containing the space gap between the lead and ego vehicle at time (t). 
        N (int): Number of timesteps that the simulation will run for (900).
        dt (float): Used measured frequency from (Wang et al., 2020).
        true_theta (array - length: 3): Array containing the true parameter values for alpha, beta, and tau.
        label (string): Label indicating which scenario is being simulated in this filter.  

    Returns:
    --------
        gamma_history (array): Array containing the values of each gamma entry in the vector from timestep 0:900
        theta_history (array): Array containing the values of each parameter, alpha, beta, and tau from timestep 0:900.
    """
    N = min(len(u_t), len(v_t), len(s_t))

    u_t = np.asarray(u_t, dtype=float).reshape(-1) # converts input data into a workable array
    v_t = np.asarray(v_t, dtype=float).reshape(-1) # converts input data into a workable array
    s_t = np.asarray(s_t, dtype=float).reshape(-1) # converts input data into a workable array

    gamma_est = np.array([0.976, 0.01, 0.01], dtype=float) # initial gamma_0 from paper 

    gamma_history = np.zeros((N, 3), dtype=float)
    theta_history = np.zeros((N, 3), dtype=float)

    gamma_history[0] = gamma_est
    theta_history[0] = invert_gamma(gamma_est, dt)

    S = (1.0 / 0.1) * np.eye(3, dtype=float) # intitialize matrix S for accumulation of outer products

    for k in range(0, N-1): # run from 0 to k-1 to get v_{k+1}
        xk = np.array([v_t[k], s_t[k], u_t[k]], dtype=float) 
        yk = float(v_t[k + 1])                                 

        S += np.outer(xk, xk) # update S_{k}

        try:
            P = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # If S is not linearly independent or ill-conditioned we compute the
            # Moore-Penrose pseudoinverse to gurantee an inverse for computational purposes.
            P = np.linalg.pinv(S)

        err = yk - xk.dot(gamma_est)
        gamma_est = gamma_est + (P @ xk) * err

        # Store next gamma and theta history values.
        gamma_history[k + 1] = gamma_est
        theta_history[k + 1] = invert_gamma(gamma_est, dt) # recovers final parameters per iteration

    alpha_est_final, beta_est_final, tau_est_final = theta_history[-1]

    print(f"\n[SCENARIO: {label}]")
    print("Final estimated alpha = %.3f (true=%.3f)" % (alpha_est_final, true_theta[0]))
    print("Final estimated beta  = %.3f (true=%.3f)"  % (beta_est_final,  true_theta[1]))
    print("Final estimated tau   = %.3f (true=%.3f)"  % (tau_est_final,   true_theta[2]))
    print("-------------------------")
    print(f"Alpha Error: {true_theta[0] - alpha_est_final:.3f}")
    print(f"Beta  Error: {true_theta[1] - beta_est_final:.3f}")
    print(f"Tau   Error: {true_theta[2] - tau_est_final:.3f}")

    # Check for condition number of matrix A, as a base metric of numerical stability
    X_full = np.stack([v_t[:-1], s_t[:-1], u_t[:-1]], axis=1).astype(float)
    cond_num = np.linalg.cond(X_full.T @ X_full)
    print(f"[SCENARIO: {label}] cond(X^T X) = {cond_num:.2e}")

    return gamma_history, theta_history

def particle_filter(u_t, v_t, s_t, dt, true_theta, label, Np=500, mu_xa0=None, Q0=None, Q=None, R=None):
    """
    Executes the particle filter algorithm for parameter estimation of ACC vehicle systems.

    Args:
    -----
        u_t (array - length: 900): Array containing velocity values at time (t) of the leading vehicle.
        v_t (array - length: 900): Array containing velocity values at time (t) of the following (ego) vehicle.
        s_t (array - length: 900): Array containing the space gap between the lead and ego vehicle at time (t).
        dt (float): Used measured frequency from (Wang et al., 2020).
        true_theta (array - length: 3): Array containing the true parameter values for alpha, beta, and tau.
        label (string): Label indicating which scenario is being simulated in this filter.
        Np (int, optional): Number of particles used in the filter. Defaults to 500 from paper.
        mu_xa0 (array - length: 5, optional): Initial augmented state mean vector [s0, v0, alpha_0, beta_0, tau_0]. Defaults to paper values from Table I.
        Q0 (array - shape: 5x5, optional): Initial covariance matrix for particle initialization. Defaults to diag[0.5, 0.5, 0.2, 0.2, 0.3]^2 from paper.
        Q (array - shape: 5x5, optional): Process noise covariance matrix. Defaults to diag[0.2, 0.1, 0.01, 0.01, 0.01]^2 from paper.
        R (array - shape: 2x2, optional): Measurement noise covariance matrix for [s, v] observations. Defaults to diag[0.2, 0.1]^2 from paper.

    Returns:
    --------
        theta_history (array): Array of shape (N, 3) containing the MAP estimates of [alpha, beta, tau] at each timestep.
        theta_mean_history (array): Array of shape (N, 3) containing the weighted mean estimates of [alpha, beta, tau] at each timestep.
        theta_std_history (array): Array of shape (N, 3) containing the weighted standard deviation of [alpha, beta, tau] at each timestep.
        particles_snapshots (dict): Dictionary mapping timestep indices to particle arrays of shape (Np, 3) at snapshot times for PDF plotting.
    
    Algorithm:
    ----------
        1. Initialize parameters and particle set from mu_xa0 and Q0.
        2. Propagate particles through CTH-RV dynamics (Equation 12).
        3. Add process noise and enforce physical constraints.
        4. Update particle weights using Gaussian likelihood of measurement.
        5. Normalize weights; reinitialize uniformly if all weights collapse.
        6. Resample via systematic resampling when effective sample size drops below Np/2.
        7. Store MAP, mean, and std estimates of [alpha, beta, tau] at each timestep.
    """
    
    # Get data length
    N = min(len(u_t), len(v_t), len(s_t))
    u_t = np.asarray(u_t, dtype=float).reshape(-1)
    v_t = np.asarray(v_t, dtype=float).reshape(-1)
    s_t = np.asarray(s_t, dtype=float).reshape(-1)
    
    # Default parameters from Table I in the paper
    if mu_xa0 is None:
        # Initial augmented state mean: [s0, v0, α0, β0, τ0]
        # Paper uses: [37.8, 32.5, 0.1, 0.1, 1.4]
        mu_xa0 = np.array([s_t[0], v_t[0], 0.1, 0.1, 1.4])
    
    if Q0 is None:
        # Initial covariance (diagonal): paper uses diag[0.5, 0.5, 0.2, 0.2, 0.3]^2
        Q0 = np.diag([0.5, 0.5, 0.2, 0.2, 0.3])**2
    
    if Q is None:
        # Process noise covariance: paper uses diag[0.2, 0.1, 0.01, 0.01, 0.01]^2
        Q = np.diag([0.2, 0.1, 0.01, 0.01, 0.01])**2
    
    if R is None:
        # Measurement noise covariance: paper uses diag[0.2, 0.1]^2
        R = np.diag([0.2, 0.1])**2
    
    # Measurement matrix C (Equation 13): we observe s and v, not parameters
    C = np.array([
        [1, 0, 0, 0, 0],  # s measurement
        [0, 1, 0, 0, 0]   # v measurement
    ])

    particles = np.random.multivariate_normal(mu_xa0, Q0, size=Np)
    
    # Ensure physical constraints for space gap, velocity, and parameters.
    # Gaussian distribution can draw negative values occasionally, these contraints
    # ensure we get values that satisfy realistic conditions.
    particles[:, 0] = np.maximum(particles[:, 0], 1.0)    # s > 0
    particles[:, 1] = np.maximum(particles[:, 1], 0.0)    # v >= 0
    particles[:, 2] = np.maximum(particles[:, 2], 0.001)  # α > 0
    particles[:, 3] = np.maximum(particles[:, 3], 0.001)  # β > 0
    particles[:, 4] = np.maximum(particles[:, 4], 0.1)    # τ > 0
    
    # Initialize equal weights
    weights = np.ones(Np) / Np
    
    # Storage for history
    theta_history = np.zeros((N, 3))       # MAP estimates [α, β, τ]
    theta_mean_history = np.zeros((N, 3))  # Mean estimates
    theta_std_history = np.zeros((N, 3))   # Std of estimates
    
    # Store initial estimates
    theta_history[0] = particles[np.argmax(weights), 2:5]
    theta_mean_history[0] = np.mean(particles[:, 2:5], axis=0)
    theta_std_history[0] = np.std(particles[:, 2:5], axis=0)
    
    # Plotting features
    desired_secs = [0, 200, 400, 600, 900]
    snapshot_times = []
    for sec in desired_secs:
        idx = int(sec / dt)
        if idx >= N:
            idx = N - 1
        snapshot_times.append(idx)
    # ensure the first and last are present and unique
    snapshot_times = sorted(set([0] + snapshot_times + [N - 1]))
    particles_snapshots = {0: particles[:, 2:5].copy()}
    
    # Process noise standard deviations (for adding noise to particles)
    Q_std = np.sqrt(np.diag(Q))
    R_std = np.sqrt(np.diag(R))
    
    # MAIN LOOP: k = 1 to N-1
    for k in range(1, N):
        
        # STATE PROPAGATION (Equation 12 in paper)
        
        # Get lead vehicle velocity at previous timestep
        u_prev = u_t[k-1]
        
        # Propagate each particle through the dynamics
        particles_new = np.zeros_like(particles)
        
        for i in range(Np):
            s_prev = particles[i, 0]
            v_prev = particles[i, 1]
            alpha_i = particles[i, 2]
            beta_i = particles[i, 3]
            tau_i = particles[i, 4]
            
            # CTH-RV dynamics (Equation 12)
            s_new = s_prev + dt * (u_prev - v_prev)
            
            # v_k = v_{k-1} + ΔT[α(s - τv) + β(u - v)]
            acc = alpha_i * (s_prev - tau_i * v_prev) + beta_i * (u_prev - v_prev)
            v_new = v_prev + dt * acc
            
            # Parameters remain constant (random walk with small noise)
            alpha_new = alpha_i
            beta_new = beta_i
            tau_new = tau_i
            
            particles_new[i] = [s_new, v_new, alpha_new, beta_new, tau_new]
        
        # Add process noise w_k ~ N(0, Q)
        process_noise = np.random.randn(Np, 5) * Q_std
        particles_new += process_noise
        
        # Enforce physical constraints
        particles_new[:, 0] = np.maximum(particles_new[:, 0], 0.1)    # s > 0
        particles_new[:, 1] = np.maximum(particles_new[:, 1], 0.0)    # v >= 0
        particles_new[:, 2] = np.maximum(particles_new[:, 2], 0.001)  # α > 0
        particles_new[:, 3] = np.maximum(particles_new[:, 3], 0.001)  # β > 0
        particles_new[:, 4] = np.maximum(particles_new[:, 4], 0.1)    # τ > 0
        
        particles = particles_new
        
        # STATE UPDATE (Equation 16 in paper)
        # Current measurement: y_k = [s_k, v_k]^T (from data)
        y_k = np.array([s_t[k], v_t[k]])
        
        # Compute likelihood for each particle
        for i in range(Np):
            # Predicted measurement from particle
            y_pred = C @ particles[i]  # [s_pred, v_pred]
            
            # Innovation (measurement residual)
            innovation = y_k - y_pred
            
            # Likelihood (Gaussian): p(y|x) ∝ exp(-0.5 * innovation^T R^{-1} innovation)
            # Using log-likelihood for numerical stability
            likelihood = np.exp(-0.5 * innovation @ np.linalg.inv(R) @ innovation)
            
            # Update weight
            weights[i] *= likelihood
        
        # NORMALIZE WEIGHTS
        weight_sum = np.sum(weights)
        if weight_sum > 1e-300:  # Avoid division by zero
            weights = weights / weight_sum
        else:
            # If all weights are essentially zero, reinitialize
            weights = np.ones(Np) / Np
        
        # RESAMPLE (Systematic Resampling)
        # Draw particles with probability proportional to weights
        N_eff = 1.0 / np.sum(weights**2)
        
        # AFTER
        indices = systematic_resample(weights)
        particles = particles[indices]
        weights = np.ones(Np) / Np  # weights reset per Algorithm 1
        
        # STORE ESTIMATES:
        
        # MAP estimate (particle with highest weight)
        map_idx = np.argmax(weights)
        theta_history[k] = particles[map_idx, 2:5]
        
        # Mean and std of posterior
        theta_mean_history[k] = np.average(particles[:, 2:5], axis=0, weights=weights)
        
        # Weighted standard deviation
        theta_var = np.average((particles[:, 2:5] - theta_mean_history[k])**2, 
                               axis=0, weights=weights)
        theta_std_history[k] = np.sqrt(theta_var)
        
        # Store snapshots for plotting PDFs
        if k in snapshot_times:
            particles_snapshots[k] = particles[:, 2:5].copy()
    
    alpha_est, beta_est, tau_est = theta_mean_history[-1]
    
    print(f"\n[PARTICLE FILTER - SCENARIO: {label}]")
    print("=" * 50)
    print(f"Number of particles: {Np}")
    print("-" * 50)
    print("Final estimated α = %.4f (true=%.4f)" % (alpha_est, true_theta[0]))
    print("Final estimated β = %.4f (true=%.4f)" % (beta_est, true_theta[1]))
    print("Final estimated τ = %.4f (true=%.4f)" % (tau_est, true_theta[2]))
    print("-" * 50)
    print(f"α Error: {abs(true_theta[0] - alpha_est):.4f}")
    print(f"β Error: {abs(true_theta[1] - beta_est):.4f}")
    print(f"τ Error: {abs(true_theta[2] - tau_est):.4f}")
    print("=" * 50)
    
    return theta_history, theta_mean_history, theta_std_history, particles_snapshots


def systematic_resample(weights):
    """
    Performs systematic resampling to draw particle indices proportional to their weights.

    Args:
    -----
        weights (array - length: Np): Normalized particle weights summing to 1.

    Returns:
    --------
        indices (array - length: Np): Resampled particle indices drawn proportionally to weights.

    Notes:
    ------
        Systematic resampling uses a single random offset and deterministically spaces
        the remaining samples, reducing variance compared to multinomial resampling.
        Refer to `pf.ipynb` for context on when resampling is triggered during the filter.
    """
    Np = len(weights)
    indices = np.zeros(Np, dtype=int)
    
    # Cumulative sum of weights
    cumsum = np.cumsum(weights)
    
    # Starting point: random offset in [0, 1/Np)
    u0 = np.random.uniform(0, 1.0/Np)
    
    # Systematic resampling
    j = 0
    for i in range(Np):
        u = u0 + i / Np
        while u > cumsum[j]:
            j += 1
        indices[i] = j
    
    return indices


# PLOTTING FUNCTIONS

def plot_parameter_convergence(theta_history, true_theta, dt, label, method="PF"):
    """
    Plots the convergence of estimated parameters alpha, beta, and tau over time.

    Args:
    -----
        theta_history (array - shape: Nx3): Estimated parameters [alpha, beta, tau] at each timestep.
        true_theta (array - length: 3): True parameter values [alpha, beta, tau].
        dt (float): Used measured frequency from (Wang et al., 2020).
        label (string): Label indicating which scenario is being plotted.
        method (string, optional): Name of the estimation method used in the plot title. Defaults to "PF".

    Returns:
    --------
        fig (matplotlib.figure.Figure): Figure object containing the three-panel convergence plot.
    """
    N = theta_history.shape[0]
    time_axis = np.arange(N) * dt  # Convert to seconds
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    param_names = ['α (alpha)', 'β (beta)', 'τ (tau)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (ax, name, color) in enumerate(zip(axes, param_names, colors)):
        ax.plot(time_axis, theta_history[:, i], color=color, linewidth=1.5, 
                label=f'Estimated {name}')
        ax.axhline(y=true_theta[i], color='black', linestyle='--', linewidth=2,
                   label=f'True {name} = {true_theta[i]}')
        ax.set_ylabel(name, fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_axis[-1]])
    
    axes[-1].set_xlabel('Time [s]', fontsize=12)
    fig.suptitle(f'{method} Parameter Convergence - {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_parameter_pdfs(particles_snapshots, true_theta, dt, label):
    """
    Plots the posterior PDFs of parameters alpha, beta, and tau at multiple time snapshots.

    Args:
    -----
        particles_snapshots (dict): Dictionary mapping timestep indices to particle arrays of
            shape (Np, 3) containing [alpha, beta, tau] values at each snapshot time.
        true_theta (array - length: 3): True parameter values [alpha, beta, tau].
        dt (float): Used measured frequency from (Wang et al., 2020).
        label (string): Label indicating which scenario is being plotted.

    Returns:
    --------
        fig (matplotlib.figure.Figure): Figure object containing the three-panel PDF plot.

    Notes:
    ------
        Replicates Figures 2 and 3 from (Wang et al., 2020). PDFs are approximated
        via normalized histograms of the particle distribution at each snapshot time.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    
    param_names = ['α', 'β', 'τ']
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(particles_snapshots)))
    
    sorted_times = sorted(particles_snapshots.keys())
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        for j, t in enumerate(sorted_times):
            particles = particles_snapshots[t]
            time_label = f'k = {int(t * dt)}s'
            
            # Compute histogram/KDE
            data = particles[:, i]
            
            # Use histogram for PDF approximation
            counts, bins = np.histogram(data, bins=30, density=True)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            
            ax.plot(bin_centers, counts, color=colors[j], linewidth=1.5, 
                    label=time_label, alpha=0.8)
        
        ax.axvline(x=true_theta[i], color='black', linestyle='--', linewidth=2,
                   label='True value')
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Posterior Parameter PDFs - {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_convergence_with_uncertainty(theta_mean_history, theta_std_history,
                                       true_theta, dt, label):
    """
    Plots parameter convergence over time with +/- 1 standard deviation uncertainty bands.

    Args:
    -----
        theta_mean_history (array - shape: Nx3): Weighted mean estimates of [alpha, beta, tau] at each timestep.
        theta_std_history (array - shape: Nx3): Weighted standard deviations of [alpha, beta, tau] at each timestep.
        true_theta (array - length: 3): True parameter values [alpha, beta, tau].
        dt (float): Used measured frequency from (Wang et al., 2020).
        label (string): Label indicating which scenario is being plotted.

    Returns:
    --------
        fig (matplotlib.figure.Figure): Figure object containing the three-panel convergence plot with shaded uncertainty bands.
    """
    N = theta_mean_history.shape[0]
    time_axis = np.arange(N) * dt
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    param_names = ['α (alpha)', 'β (beta)', 'τ (tau)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (ax, name, color) in enumerate(zip(axes, param_names, colors)):
        mean = theta_mean_history[:, i]
        std = theta_std_history[:, i]
        
        ax.plot(time_axis, mean, color=color, linewidth=2, label=f'Mean estimate')
        ax.fill_between(time_axis, mean - std, mean + std, 
                        color=color, alpha=0.3, label='±1 std')
        ax.axhline(y=true_theta[i], color='black', linestyle='--', linewidth=2,
                   label=f'True = {true_theta[i]}')
        
        ax.set_ylabel(name, fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_axis[-1]])
    
    axes[-1].set_xlabel('Time [s]', fontsize=12)
    fig.suptitle(f'PF Parameter Convergence with Uncertainty - {label}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def compare_rls_pf(u_t, v_t, s_t, dt, true_theta, label, Np=500):
    """
    Runs both the RLS filter and particle filter on the same trajectory data and plots a side-by-side comparison.

    Args:
    -----
        u_t (array - length: 900): Array containing velocity values at time (t) of the leading vehicle.
        v_t (array - length: 900): Array containing velocity values at time (t) of the following (ego) vehicle.
        s_t (array - length: 900): Array containing the space gap between the lead and ego vehicle at time (t).
        dt (float): Used measured frequency from (Wang et al., 2020).
        true_theta (array - length: 3): Array containing the true parameter values for alpha, beta, and tau.
        label (string): Label indicating which scenario is being simulated.
        Np (int, optional): Number of particles used in the particle filter. Defaults to 500.

    Returns:
    --------
        fig (matplotlib.figure.Figure): Figure object containing the RLS vs PF comparison plot.
        theta_hist_rls (array - shape: Nx3): RLS parameter estimates [alpha, beta, tau] at each timestep.
        theta_mean_pf (array - shape: Nx3): Particle filter weighted mean estimates of [alpha, beta, tau] at each timestep.
        particles_snap (dict): Dictionary of particle snapshots at selected timesteps from the particle filter.
    """
    N = len(u_t)
    
    # Run RLS
    print("\n" + "="*60)
    print("Running RLS...")
    print("="*60)
    gamma_hist_rls, theta_hist_rls = rls_filter(u_t, v_t, s_t, N, dt, true_theta, label)
    
    # Run PF
    print("\n" + "="*60)
    print("Running Particle Filter...")
    print("="*60)
    theta_hist_pf, theta_mean_pf, theta_std_pf, particles_snap = particle_filter(
        u_t, v_t, s_t, dt, true_theta, label, Np=Np
    )
    
    # Plot comparison
    time_axis = np.arange(N) * dt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    param_names = ['α (alpha)', 'β (beta)', 'τ (tau)']
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        # RLS estimate
        ax.plot(time_axis, theta_hist_rls[:, i], 'b-', linewidth=1.5, 
                label='RLS', alpha=0.8)
        
        # PF mean estimate with uncertainty
        ax.plot(time_axis, theta_mean_pf[:, i], 'r-', linewidth=1.5, 
                label='PF (mean)')
        ax.fill_between(time_axis, 
                        theta_mean_pf[:, i] - theta_std_pf[:, i],
                        theta_mean_pf[:, i] + theta_std_pf[:, i],
                        color='red', alpha=0.2, label='PF ±1 std')
        
        # True value
        ax.axhline(y=true_theta[i], color='black', linestyle='--', 
                   linewidth=2, label=f'True = {true_theta[i]}')
        
        ax.set_ylabel(name, fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_axis[-1]])
    
    axes[-1].set_xlabel('Time [s]', fontsize=12)
    fig.suptitle(f'RLS vs Particle Filter Comparison - {label}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, theta_hist_rls, theta_mean_pf, particles_snap