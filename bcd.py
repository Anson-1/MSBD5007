import numpy as np
import time
from data import compute_objective, compute_accuracy

def bcd_svm(X, y, X_test, y_test, lambda_=1.0, max_epochs=1000, tol=1e-3, random_seed=42):
    rng = np.random.default_rng(random_seed)
    n, d = X.shape

    init_start_time = time.time()
    alpha = np.zeros(n)
    b = 0.0
    K = X @ X.T
    total_opt_time = time.time() - init_start_time

    obj_history, acc_history, time_history = [], [], []

    def reconstruct_primal():
        w_curr = (alpha * y) @ X
        return w_curr.reshape(-1, 1), b

    w_init, b_init = reconstruct_primal()
    obj_history.append(compute_objective(w_init, b_init, X, y, lambda_))
    acc_history.append(compute_accuracy(w_init, b_init, X_test, y_test))
    time_history.append(total_opt_time)

    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        num_changed_alphas = 0

        for i in range(n):
            f_i = np.sum(alpha * y * K[:, i]) + b
            E_i = f_i - y[i]

            # Fast random j selection (avoids O(n) list creation)
            j = rng.integers(0, n - 1)
            if j >= i:
                j += 1

            f_j = np.sum(alpha * y * K[:, j]) + b
            E_j = f_j - y[j]

            alpha_i_old, alpha_j_old = alpha[i], alpha[j]

            if y[i] != y[j]:
                L = max(0, alpha[j] - alpha[i])
                H = min(lambda_, lambda_ + alpha[j] - alpha[i])
            else:
                L = max(0, alpha[i] + alpha[j] - lambda_)
                H = min(lambda_, alpha[i] + alpha[j])

            if L == H:
                continue

            eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
            if eta >= 0:
                continue

            alpha_j_new = alpha_j_old - (y[j] * (E_i - E_j)) / eta

            if alpha_j_new > H:
                alpha_j_new = H
            elif alpha_j_new < L:
                alpha_j_new = L

            if abs(alpha_j_new - alpha_j_old) < 1e-5:
                continue

            alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)

            b1 = b - E_i - y[i] * (alpha_i_new - alpha_i_old) * K[i, i] - y[j] * (alpha_j_new - alpha_j_old) * K[j, i]
            b2 = b - E_j - y[i] * (alpha_i_new - alpha_i_old) * K[i, j] - y[j] * (alpha_j_new - alpha_j_old) * K[j, j]

            if 0 < alpha_i_new < lambda_:
                b = b1
            elif 0 < alpha_j_new < lambda_:
                b = b2
            else:
                b = (b1 + b2) / 2.0

            alpha[i], alpha[j] = alpha_i_new, alpha_j_new
            num_changed_alphas += 1

        total_opt_time += (time.time() - epoch_start_time)

        # Metrics computed outside the timer
        w_curr, b_curr = reconstruct_primal()
        obj_history.append(compute_objective(w_curr, b_curr, X, y, lambda_))
        acc_history.append(compute_accuracy(w_curr, b_curr, X_test, y_test))
        time_history.append(total_opt_time)

        if num_changed_alphas == 0:
            break

    w_final, b_final = reconstruct_primal()
    return w_final, b_final, obj_history, acc_history, time_history, epoch + 1
