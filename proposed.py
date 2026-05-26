import numpy as np
import time
from data import compute_objective, compute_accuracy

def proposed_svm(X, y, X_test, y_test, lambda_=1.0, max_iter=500, tol=1e-6, random_seed=42):
    """
    Proposed Method: Newton's Method on Huber-Smoothed Hinge Loss
    with Continuation Strategy (progressively decreasing smoothing parameter).

    Key ideas:
    - Replace non-smooth hinge loss max(0, t) with C1-smooth Huber hinge phi_mu(t)
    - Apply Newton's method (quadratic convergence) to the smooth problem
    - Use continuation: solve for large mu first, then decrease mu and warm-start
    - Final mu is very small, so solution closely approximates the true SVM

    Huber hinge phi_mu(t):
      phi_mu(t) = 0            if t <= 0
      phi_mu(t) = t^2 / (2*mu) if 0 < t <= mu
      phi_mu(t) = t - mu/2     if t > mu

    Gradient phi_mu'(t):
      = 0     if t <= 0
      = t/mu  if 0 < t <= mu
      = 1     if t > mu

    Hessian contribution phi_mu''(t):
      = 0     if t <= 0
      = 1/mu  if 0 < t <= mu
      = 0     if t > mu
    """
    n, d = X.shape

    # Continuation schedule: decreasing smoothing parameters
    mu_schedule = [1.0, 0.1, 0.01, 0.001]

    init_start_time = time.time()
    w = np.zeros((d, 1))
    b = 0.0
    total_opt_time = time.time() - init_start_time

    obj_history, acc_history, time_history = [], [], []

    # Record initial state
    obj_history.append(compute_objective(w, b, X, y, lambda_))
    acc_history.append(compute_accuracy(w, b, X_test, y_test))
    time_history.append(total_opt_time)

    y_col = y.reshape(-1, 1)
    total_iters = 0

    for mu in mu_schedule:
        for it in range(max_iter):
            iter_start = time.time()

            # Compute residuals: r_i = 1 - y_i * (x_i^T w + b)
            margins = X @ w + b
            r = 1 - y_col * margins  # (n, 1)

            # Classify points into three zones
            inactive = (r <= 0).ravel()        # hinge not active
            quadratic = ((r > 0) & (r <= mu)).ravel()  # smooth transition zone
            active = (r > mu).ravel()          # fully active hinge

            # Compute gradient of smoothed objective
            # grad_w = w + lambda * sum of (-y_i * x_i) * phi'(r_i)
            # grad_b = lambda * sum of (-y_i) * phi'(r_i)
            phi_prime = np.zeros((n, 1))
            phi_prime[quadratic] = r[quadratic] / mu
            phi_prime[active] = 1.0

            grad_w = w + lambda_ * (X.T @ (-y_col * phi_prime))
            grad_b = lambda_ * np.sum(-y_col * phi_prime)

            grad = np.vstack([grad_w, [[grad_b]]])
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tol:
                total_opt_time += (time.time() - iter_start)
                break

            # Compute Hessian of smoothed objective
            # H_ww = I + (lambda/mu) * X_S^T X_S  (S = quadratic zone)
            # H_wb = (lambda/mu) * X_S^T * ones
            # H_bb = (lambda/mu) * |S|
            X_S = X[quadratic]  # rows in quadratic zone
            n_S = X_S.shape[0]

            H_ww = np.eye(d) + (lambda_ / mu) * (X_S.T @ X_S)
            H_wb = (lambda_ / mu) * X_S.T @ np.ones((n_S, 1))
            H_bb = (lambda_ / mu) * n_S

            # Build full (d+1) x (d+1) Hessian
            H = np.block([
                [H_ww, H_wb],
                [H_wb.T, np.array([[H_bb]])]
            ])

            # Add small regularization for numerical stability when S is empty
            H += 1e-12 * np.eye(d + 1)

            # Newton direction: delta = -H^{-1} * grad
            delta = np.linalg.solve(H, -grad)
            delta_w = delta[:d]
            delta_b = delta[d, 0]

            # Backtracking line search (Armijo condition)
            alpha = 1.0
            c_armijo = 1e-4
            current_obj = _smoothed_objective(w, b, X, y_col, lambda_, mu)
            directional_deriv = grad.T @ delta

            for _ in range(30):
                w_new = w + alpha * delta_w
                b_new = b + alpha * delta_b
                new_obj = _smoothed_objective(w_new, b_new, X, y_col, lambda_, mu)
                if new_obj <= current_obj + c_armijo * alpha * directional_deriv[0, 0]:
                    break
                alpha *= 0.5

            w = w + alpha * delta_w
            b = b + alpha * delta_b

            total_opt_time += (time.time() - iter_start)
            total_iters += 1

            # Record metrics
            obj_history.append(compute_objective(w, b, X, y, lambda_))
            acc_history.append(compute_accuracy(w, b, X_test, y_test))
            time_history.append(total_opt_time)

    return w, b, obj_history, acc_history, time_history, total_iters


def _smoothed_objective(w, b, X, y_col, lambda_, mu):
    """Compute the Huber-smoothed SVM objective."""
    margins = X @ w + b
    r = 1 - y_col * margins

    loss = np.zeros_like(r)
    quad_mask = (r > 0) & (r <= mu)
    active_mask = r > mu
    loss[quad_mask] = r[quad_mask]**2 / (2 * mu)
    loss[active_mask] = r[active_mask] - mu / 2

    return 0.5 * np.linalg.norm(w)**2 + lambda_ * np.sum(loss)
