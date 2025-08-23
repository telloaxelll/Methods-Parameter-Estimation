# Import Needed Dependencies: 
import numpy as np
import matplotlib.pyplot as plt
import os

plot_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plot_dir, exist_ok=True)

"""
This functions.py file will host all of the necessary functions to execute 
functions of the Recursive Least Squares. 
"""

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
        tau = 0.0
    else:
        tau = ((1 - gamma1 - gamma3) / gamma2)
    return alpha, beta, tau

# RLS Filter Function: 
# note: need to remove scenario paramter from main definition since only scenario 1
# uses that particular parameter. 
def rls_filter(u_t, v_t, s_t, time, dt, true_theta, label):
    """
    Given u_t, v_t, s_t, time, and dt:
    - We will set initial guess for theta vector
    - Initialize covariance matrix
    - Allocate estimation tracking arrays for recursive updates
    RLS: 
    - Concatenate all of u_t, v_t, s_t into one matrix X
    - Initialize v_t into Y as the output vector
    - Compute Kalman gain
    - Update parameters at each step
    - Update covariance matrix
    """
    gamma_est = np.array([0.9, 0.01, 0.01])  # some initial guess
    P = np.eye(3) * 1000.0 # covariance matrix 

    gamma_history = np.zeros((time, 3))
    theta_history = np.zeros((time, 3))  # [alpha, beta, tau] at each step - 3 x 900 matrix storing all values of theta

    # Initialize RLS:
    gamma_history[0] = gamma_est
    theta_history[0] = invert_gamma(gamma_est, dt)

    # RLS Algorithm:
    for k in range(1, time):
        ''' 
        Debug Issue: 
        - We know that matrix is ill-conditioned for two scenarios
          however, it's showing us that matrix X has a condition number of 
          1. Which cannot simply be since previous numerical results showed that overall data 
          is not as close which can give us errors. 

        TO-DO: 
        - Configure via a sanity check on the code to ensure that condition number is being
          adequately reflected based on the ill conditions of the matrix X. 
        '''

        X = np.array([v_t[k-1], s_t[k-1], u_t[k-1]])

        #X = np.array([v_t[k-1], s_t[k-1], u_t[k-1]])

        y = v_t[k]

        # Compute Kalman gain
        denominator = 1.0 + X.T.dot(P).dot(X)
        K = P.dot(X) / denominator

        # RLS Update for Gamma Parameters:
        gamma_est = gamma_est + K * (y - X.T.dot(gamma_est))
        P = (np.eye(3) - np.outer(K, X)).dot(P)

        # Store History of Parameter Estimates:
        gamma_history[k] = gamma_est
        theta_history[k] = invert_gamma(gamma_est, dt)

    # Print Results
    alpha_est_final, beta_est_final, tau_est_final = theta_history[-1]
    print(f"\n[SCENARIO: {label}]")
    print("Final estimated alpha = %.3f (true=%.3f)" % (alpha_est_final, true_theta[0]))
    print("Final estimated beta  = %.3f (true=%.3f)"  % (beta_est_final, true_theta[1]))
    print("Final estimated tau   = %.3f (true=%.3f)"   % (tau_est_final, true_theta[2]))
    print("-------------------------")
    print(f"Alpha Error: {true_theta[0] - alpha_est_final:.3f}")
    print(f"Beta Error: {true_theta[1] - beta_est_final:.3f}")
    print(f"Tau Error: {true_theta[2] - tau_est_final:.3f}")

    # Plot - Alpha, Beta, Tau Convergence
    t_axis = np.arange(time)
    fig, axes = plt.subplots(3,1, figsize=(12,10), sharex=True)

    params  = ["alpha", "beta", "tau"]
    trueval = [true_theta[0], true_theta[1], true_theta[2]]
    colors  = ["r", "g", "b"]

    for i, ax in enumerate(axes):
        ax.plot(t_axis, theta_history[:, i], label=f"Estimated {params[i]}", color=colors[i])
        ax.axhline(y=trueval[i], color=colors[i], linestyle="--", label=f"True {params[i]}")
        ax.legend()
        ax.grid()

    axes[-1].set_xlabel("Time step (k)")
    plt.suptitle(f"RLS Parameter Convergence: Scenario {label}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"Convergence_Parameters_Scenario_{label}.png"
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # Stack the full design matrix X using all past samples
    X_full = np.stack([v_t[:-1], s_t[:-1], u_t[:-1]], axis=1)  # shape: (time-1, 3)

    # Compute condition number
    condition_number = np.linalg.cond(X_full)

    print(f"[SCENARIO: {label}] Condition Number of X: {condition_number:.2e}")