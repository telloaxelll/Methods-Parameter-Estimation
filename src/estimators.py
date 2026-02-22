import numpy as np
import matplotlib.pyplot as plt
import os

# Invert Gamma Function:  
def invert_gamma(gamma, dt):
    """
    Given gamma1, gamma2, gamma3 = alpha, beta, tau, we are able to directly
    solve for alpha, beta, tau by using the following equations algebraically:
    - alpha = gamma2 / dt
    - beta = gamma3 / dt
    - tau = ((1 - gamma1 - gamma3) / gamma2)
    """
    gamma1, gamma2, gamma3 = gamma

    alpha = gamma2/dt
    beta  = gamma3/dt

    if abs(alpha) < 1e-8: # Added tolerance for numerical stability
        # Add print statement for debugging
        tau = 0.0
    else:
        tau = ((1 - gamma1 - gamma3) / gamma2)
    return alpha, beta, tau


def rls_filter(u_t, v_t, s_t, N, dt, true_theta, label):
    # Length from the data (robust to 'time' mismatches)
    N = min(len(u_t), len(v_t), len(s_t))
    u_t = np.asarray(u_t, dtype=float).reshape(-1)
    v_t = np.asarray(v_t, dtype=float).reshape(-1)
    s_t = np.asarray(s_t, dtype=float).reshape(-1)

    # Paper parameters: γ = [γ1, γ2, γ3]^T; initial guess like the paper
    gamma_est = np.array([0.976, 0.01, 0.01], dtype=float)

    # Histories (store at index k+1 so -1 is the final estimate)
    gamma_history = np.zeros((N, 3), dtype=float)
    theta_history = np.zeros((N, 3), dtype=float)
    gamma_history[0] = gamma_est
    theta_history[0] = invert_gamma(gamma_est, dt)

    # Cumulative outer-product S_k = sum_{i=0}^k x_i x_i^T  (paper’s P_k^{-1})
    S = np.zeros((3, 3), dtype=float)

    for k in range(0, N - 1):  # we need v_{k+1}
        xk = np.array([v_t[k], s_t[k], u_t[k]], dtype=float)  # X_k row as a 1D vector
        yk = float(v_t[k + 1])                                 # Y_k = v_{k+1}

        # Update S_k
        S += np.outer(xk, xk)

        # P_k = S_k^{-1}; use pinv when singular (still exact LS at this step)
        try:
            P = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(S)

        # γ̂_k = γ̂_{k-1} + P_k x_k (y_k - x_k^T γ̂_{k-1})
        err = yk - xk.dot(gamma_est)
        gamma_est = gamma_est + (P @ xk) * err

        # store
        gamma_history[k + 1] = gamma_est
        theta_history[k + 1] = invert_gamma(gamma_est, dt)

    alpha_est_final, beta_est_final, tau_est_final = theta_history[-1]
    print(f"\n[SCENARIO: {label}]")
    print("Final estimated alpha = %.3f (true=%.3f)" % (alpha_est_final, true_theta[0]))
    print("Final estimated beta  = %.3f (true=%.3f)"  % (beta_est_final,  true_theta[1]))
    print("Final estimated tau   = %.3f (true=%.3f)"  % (tau_est_final,   true_theta[2]))
    print("-------------------------")
    print(f"Alpha Error: {true_theta[0] - alpha_est_final:.3f}")
    print(f"Beta  Error: {true_theta[1] - beta_est_final:.3f}")
    print(f"Tau   Error: {true_theta[2] - tau_est_final:.3f}")

    # Optional condition number check of the batch X used in the paper’s (7)
    X_full = np.stack([v_t[:-1], s_t[:-1], u_t[:-1]], axis=1).astype(float)
    cond_num = np.linalg.cond(X_full.T @ X_full)
    print(f"[SCENARIO: {label}] cond(X^T X) = {cond_num:.2e}")

    # Return histories if you want to plot outside
    return gamma_history, theta_history


# =============================================================================
# PARTICLE FILTER IMPLEMENTATION (Algorithm 1 from Wang et al. 2020)
# =============================================================================

def particle_filter(u_t, v_t, s_t, dt, true_theta, label, 
                    Np=500, 
                    mu_xa0=None, 
                    Q0=None, 
                    Q=None, 
                    R=None):
    """
    Particle Filter for joint state and parameter estimation of CTH-RV model.
    
    Following Algorithm 1 from Wang et al. (2020):
    "Online parameter estimation methods for adaptive cruise control systems"
    
    Augmented State: x_a = [s, v, α, β, τ]^T  (Equation 10 in paper)
    
    Parameters:
    -----------
    u_t : array
        Lead vehicle velocity time series
    v_t : array  
        Following vehicle velocity time series (used as measurements)
    s_t : array
        Space gap time series (used as measurements)
    dt : float
        Time step (ΔT in the paper)
    true_theta : array
        True parameters [α, β, τ] for comparison
    label : str
        Scenario label for printing
    Np : int
        Number of particles (default 500, as in paper Table I)
    mu_xa0 : array, optional
        Initial mean of augmented state [s0, v0, α0, β0, τ0]
    Q0 : array, optional
        Initial covariance diagonal for augmented state
    Q : array, optional
        Process noise covariance diagonal
    R : array, optional
        Measurement noise covariance diagonal
        
    Returns:
    --------
    theta_history : array (N, 3)
        Estimated parameters [α, β, τ] at each timestep (MAP estimates)
    theta_mean_history : array (N, 3)
        Mean of parameter posterior at each timestep
    theta_std_history : array (N, 3)
        Std of parameter posterior at each timestep
    particles_history : list
        Full particle states at selected timesteps for plotting
    """
    
    # Get data length
    N = min(len(u_t), len(v_t), len(s_t))
    u_t = np.asarray(u_t, dtype=float).reshape(-1)
    v_t = np.asarray(v_t, dtype=float).reshape(-1)
    s_t = np.asarray(s_t, dtype=float).reshape(-1)
    
    # -----------------------------------------------------------------
    # Default parameters from Table I in the paper
    # -----------------------------------------------------------------
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
    # y = C @ x_a, where y = [s, v]^T
    C = np.array([
        [1, 0, 0, 0, 0],  # s measurement
        [0, 1, 0, 0, 0]   # v measurement
    ])
    
    # -----------------------------------------------------------------
    # INITIALIZATION (k = 0)
    # Draw Np particles from initial distribution p(x_a_0)
    # Assign equal weights ω_i = 1/Np
    # -----------------------------------------------------------------
    
    # Draw particles from multivariate normal with mean mu_xa0 and covariance Q0
    particles = np.random.multivariate_normal(mu_xa0, Q0, size=Np)  # Shape: (Np, 5)
    
    # Ensure physical constraints on initial particles
    particles[:, 0] = np.maximum(particles[:, 0], 1.0)    # s > 0
    particles[:, 1] = np.maximum(particles[:, 1], 0.0)    # v >= 0
    particles[:, 2] = np.maximum(particles[:, 2], 0.001)  # α > 0
    particles[:, 3] = np.maximum(particles[:, 3], 0.001)  # β > 0
    particles[:, 4] = np.maximum(particles[:, 4], 0.1)    # τ > 0
    
    # Initialize equal weights
    weights = np.ones(Np) / Np
    
    # -----------------------------------------------------------------
    # Storage for history
    # -----------------------------------------------------------------
    theta_history = np.zeros((N, 3))       # MAP estimates [α, β, τ]
    theta_mean_history = np.zeros((N, 3))  # Mean estimates
    theta_std_history = np.zeros((N, 3))   # Std of estimates
    
    # Store initial estimates
    theta_history[0] = particles[np.argmax(weights), 2:5]
    theta_mean_history[0] = np.mean(particles[:, 2:5], axis=0)
    theta_std_history[0] = np.std(particles[:, 2:5], axis=0)
    
    # Store particles at specific times for PDF plotting (like Figure 2/3 in paper)
    snapshot_times = [0, N//5, 2*N//5, 3*N//5, N-1]  # 0s, 200s, 400s, 600s, 900s approximately
    particles_snapshots = {0: particles[:, 2:5].copy()}
    
    # Process noise standard deviations (for adding noise to particles)
    Q_std = np.sqrt(np.diag(Q))
    R_std = np.sqrt(np.diag(R))
    
    # -----------------------------------------------------------------
    # MAIN LOOP: k = 1 to N-1
    # -----------------------------------------------------------------
    for k in range(1, N):
        
        # -------------------------------------------------------------
        # STATE PROPAGATION (Equation 12 in paper)
        # x_a_k = F_d(x_a_{k-1}, u_{k-1}) + w_k
        # -------------------------------------------------------------
        
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
            # s_k = s_{k-1} + ΔT(u_{k-1} - v_{k-1})
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
        
        # -------------------------------------------------------------
        # STATE UPDATE (Equation 16 in paper)
        # Update weights based on likelihood of measurement
        # ω_k^i = ω_{k-1}^i * p(y_k | x_a_k^i)
        # -------------------------------------------------------------
        
        # Current measurement: y_k = [s_k, v_k]^T (from data)
        y_k = np.array([s_t[k], v_t[k]])
        
        # Compute likelihood for each particle
        # p(y_k | x_a_k^i) = N(y_k; C @ x_a_k^i, R)
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
        
        # -------------------------------------------------------------
        # NORMALIZE WEIGHTS
        # ω_k^i = ω_k^i / Σ_j ω_k^j
        # -------------------------------------------------------------
        weight_sum = np.sum(weights)
        if weight_sum > 1e-300:  # Avoid division by zero
            weights = weights / weight_sum
        else:
            # If all weights are essentially zero, reinitialize
            weights = np.ones(Np) / Np
        
        # -------------------------------------------------------------
        # RESAMPLE (Systematic Resampling)
        # Draw particles with probability proportional to weights
        # -------------------------------------------------------------
        
        # Compute effective sample size to decide if resampling is needed
        N_eff = 1.0 / np.sum(weights**2)
        
        if N_eff < Np / 2:  # Resample if effective size drops below threshold
            # Systematic resampling
            indices = systematic_resample(weights)
            particles = particles[indices]
            weights = np.ones(Np) / Np  # Reset weights after resampling
        
        # -------------------------------------------------------------
        # STORE ESTIMATES
        # -------------------------------------------------------------
        
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
    
    # -----------------------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------------------
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
    Systematic resampling algorithm for particle filter.
    
    This is more efficient than multinomial resampling and produces
    lower variance estimates.
    
    Parameters:
    -----------
    weights : array
        Normalized particle weights (must sum to 1)
        
    Returns:
    --------
    indices : array
        Indices of resampled particles
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


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_parameter_convergence(theta_history, true_theta, dt, label, method="PF"):
    """
    Plot parameter convergence over time.
    
    Parameters:
    -----------
    theta_history : array (N, 3)
        Estimated parameters at each timestep
    true_theta : array
        True parameters [α, β, τ]
    dt : float
        Time step
    label : str
        Scenario label
    method : str
        Method name for title
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
    Plot posterior PDFs of parameters at different time snapshots.
    (Replicates Figure 2/3 from the paper)
    
    Parameters:
    -----------
    particles_snapshots : dict
        Dictionary mapping timestep -> particles array (Np, 3)
    true_theta : array
        True parameters [α, β, τ]
    dt : float
        Time step
    label : str
        Scenario label
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    param_names = ['α', 'β', 'τ']
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(particles_snapshots)))
    
    sorted_times = sorted(particles_snapshots.keys())
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        for j, t in enumerate(sorted_times):
            particles = particles_snapshots[t]
            time_label = f'k = {int(t*dt)}s'
            
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
    Plot parameter convergence with uncertainty bands (±1 std).
    
    Parameters:
    -----------
    theta_mean_history : array (N, 3)
        Mean parameter estimates at each timestep
    theta_std_history : array (N, 3)
        Std of parameter estimates at each timestep
    true_theta : array
        True parameters [α, β, τ]
    dt : float
        Time step
    label : str
        Scenario label
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
    Run both RLS and PF on the same data and compare results.
    
    Parameters:
    -----------
    u_t, v_t, s_t : arrays
        Velocity and gap data
    dt : float
        Time step
    true_theta : array
        True parameters
    label : str
        Scenario label
    Np : int
        Number of particles for PF
        
    Returns:
    --------
    fig : matplotlib figure
        Comparison plot
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