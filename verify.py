import numpy as np
import cvxpy as cp
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# Import YOUR exact functions from your data.py file
from data import generate_synthetic_svm_data, load_breast_cancer_binary

def run_checks(name, X_train, y_train, X_test, y_test, lambda_val=1.0):
    print(f"\n{'='*40}")
    print(f"EVALUATING: {name}")
    print(f"{'='*40}")
    
    # Flatten y just in case it's shape (n, 1) to prevent sklearn/cvxpy warnings
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    n, d = X_train.shape

    # --- CVXPY EXACT MATH CHECK ---
    print("\n--- Running CVXPY (Primal Exact Math) ---")
    w = cp.Variable(d)
    b = cp.Variable()

    # The exact primal formulation
    hinge_loss = cp.sum(cp.pos(1 - cp.multiply(y_train, X_train @ w + b)))
    objective = cp.Minimize(0.5 * cp.norm(w, 2)**2 + lambda_val * hinge_loss)

    prob = cp.Problem(objective)
    prob.solve()
    print(f"True Global Minimum Objective: {prob.value:.4f}")

    # --- SCIKIT-LEARN INDUSTRY CHECK ---
    print("\n--- Running Scikit-Learn (Industry Baselines) ---")
    C_val = lambda_val # Sklearn uses 'C' as its penalty

    # We use random_state=42 to ensure deterministic baseline results
    clf_bcd = LinearSVC(C=C_val, loss='hinge', max_iter=100000, random_state=42)
    clf_bcd.fit(X_train, y_train)
    y_pred = clf_bcd.predict(X_test)
    print(f"Sklearn LinearSVC (BCD) Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    clf_smo = SVC(C=C_val, kernel='linear', max_iter=100000, random_state=42)
    clf_smo.fit(X_train, y_train)
    y_pred_smo = clf_smo.predict(X_test)
    print(f"Sklearn SVC (SMO) Accuracy: {accuracy_score(y_test, y_pred_smo):.4f}")


# ==========================================
# 1. BREAST CANCER DATASET
# ==========================================
# Load exactly how your project does it
X_train_bc, y_train_bc, X_test_bc, y_test_bc = load_breast_cancer_binary()
run_checks("Breast Cancer Dataset", X_train_bc, y_train_bc, X_test_bc, y_test_bc, lambda_val=1.0)


# ==========================================
# 2. SYNTHETIC DATASET
# ==========================================
# Your function returns 5 things (including w_true), so we unpack the first 4
X_train_syn, y_train_syn, X_test_syn, y_test_syn, _ = generate_synthetic_svm_data()
run_checks("Synthetic Dataset", X_train_syn, y_train_syn, X_test_syn, y_test_syn, lambda_val=1.0)