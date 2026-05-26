import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def generate_synthetic_svm_data(
    n_train=1000,
    n_test=500,
    d=50,
    condition_number=100.0,
    noise_std=0.05,
    random_seed=42
):
    rng = np.random.default_rng(random_seed)
    G = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(G)
    eigenvalues = np.logspace(0, -np.log10(condition_number), d)
    Sigma_sqrt = Q @ np.diag(np.sqrt(eigenvalues))
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)

    def sample_data(n):
        Z = rng.normal(size=(n, d))
        X = Z @ Sigma_sqrt.T
        scores = X @ w_true + noise_std * rng.normal(size=n)
        y = np.where(scores >= 0, 1, -1)
        return X, y

    X_train, y_train = sample_data(n_train)
    X_test, y_test = sample_data(n_test)
    return X_train, y_train, X_test, y_test, w_true

def load_breast_cancer_binary(random_seed=42):
    data = load_breast_cancer()
    X = data.data.astype(float)

    # In sklearn, target 0 = malignant and target 1 = benign.
    # We use +1 for malignant and -1 for benign.
    y = np.where(data.target == 0, 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )

    # Standardize using training-set statistics only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid division by zero.
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test

def compute_objective(w, b, X, y, lambda_):
    y_col = y.reshape(-1, 1)
    margin = 1 - y_col * (X @ w + b)
    hinge_loss = np.maximum(0, margin)
    return 0.5 * np.linalg.norm(w)**2 + lambda_ * np.sum(hinge_loss)

def compute_accuracy(w, b, X, y):
    y_col = y.reshape(-1, 1)
    predictions = np.where(X @ w + b >= 0, 1, -1)
    return np.mean(predictions == y_col)
