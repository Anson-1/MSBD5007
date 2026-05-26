import numpy as np
import time
from data import compute_objective, compute_accuracy

def admm_svm(X, y, X_test, y_test, lambda_=1.0, gamma=1.0, max_iter=1000, tol=1e-4,
             adaptive_rho=True, mu_adapt=10.0, tau_adapt=2.0):
    n, d = X.shape
    y = y.reshape(-1, 1)
    ones = np.ones((n, 1))
    YX = y * X

    # Pre-computations (outside the main loop)
    init_start_time = time.time()
    I = np.eye(d)

    def build_lhs(gamma_val):
        return np.block([
            [I + gamma_val * (X.T @ X), gamma_val * (X.T @ ones)],
            [gamma_val * (ones.T @ X),  np.array([[gamma_val * n]])]
        ])

    LHS = build_lhs(gamma)
    w, b = np.zeros((d, 1)), 0.0
    z, mu = np.zeros((n, 1)), np.zeros((n, 1))
    total_opt_time = time.time() - init_start_time

    obj_history, acc_history, time_history = [], [], []

    for k in range(max_iter):
        iter_start_time = time.time()
        z_old = z.copy()

        # Step 1: Update (w, b)
        r = z - ones + (mu / gamma)
        RHS = np.vstack([-gamma * (X.T @ (y * r)), -gamma * (y.T @ r)])
        solution = np.linalg.solve(LHS, RHS)
        w, b = solution[:d], solution[d][0]

        # Step 2: Update z (proximal operator)
        v = ones - (YX @ w) - (b * y) - (mu / gamma)
        z = np.where(v < 0, v, np.where(v > lambda_ / gamma, v - lambda_ / gamma, 0))

        # Step 3: Update dual variable mu
        primal_residual = z + (YX @ w) + (b * y) - ones
        mu = mu + gamma * primal_residual

        # Dual residual: gamma * (z^k - z^{k-1})
        dual_residual = gamma * (z - z_old)

        # Adaptive rho update
        if adaptive_rho:
            primal_norm = np.linalg.norm(primal_residual)
            dual_norm = np.linalg.norm(dual_residual)
            if primal_norm > mu_adapt * dual_norm:
                gamma *= tau_adapt
                LHS = build_lhs(gamma)
            elif dual_norm > mu_adapt * primal_norm:
                gamma /= tau_adapt
                LHS = build_lhs(gamma)

        # Accumulate pure optimization time
        total_opt_time += (time.time() - iter_start_time)

        # Compute metrics (Timer is NOT running here)
        obj_history.append(compute_objective(w, b, X, y, lambda_))
        acc_history.append(compute_accuracy(w, b, X_test, y_test))
        time_history.append(total_opt_time)

        # Convergence check: both primal and dual residuals must be small
        if np.linalg.norm(primal_residual) < tol and np.linalg.norm(dual_residual) < tol:
            break

    return w, b, obj_history, acc_history, time_history, k+1
