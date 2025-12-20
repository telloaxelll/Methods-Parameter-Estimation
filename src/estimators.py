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