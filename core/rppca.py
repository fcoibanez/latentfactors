"""Main Risk Premium PCA Class"""
import numpy as np


class RiskPremiumPCA:
    def __init__(self, gamma, n_factors):
        self._gamma = gamma
        self._k = n_factors
        self.excess_returns = None
        self.loadings = None
        self.factors = None
        self.eigenvalues = None

    def fit(self, excess_returns, std_norm=False, var_norm=False, orthogonalize=False):
        self.excess_returns = excess_returns.copy()
        X = self.excess_returns.values
        T, N = X.shape
        _1_t = np.ones((T, T))
        I = np.eye(T)

        if std_norm:
            # Variance normalization matrix
            V = X.T @ (I - _1_t / T) @ X
            S = np.diag(np.sqrt(np.diag(V / T)))  # Diag std dev
            WN = np.inv(S)
        else:
            WN = np.eye(N)

        WT = I + self._gamma / T * _1_t

        # Generic estimator for general weighting matrices
        X_hat = X @ WN


if __name__ == "__main__":
    import pandas as pd
    import os

    crsp = pd.read_pickle(f"{os.getcwd()}/data/crsp.pkl")
    rt = crsp['excess_ret'].dropna()

    sample = pd.pivot_table(
        rt.to_frame('rt'), 
        index='date', 
        columns='permno', 
        values='rt'
    )
    sample_ids = sample.count().sort_values().tail(100).index
    sample = sample[sample_ids]

    mdl = RiskPremiumPCA(gamma=0.5, n_factors=4)
    mdl.fit(
        excess_returns=sample,
        std_norm=True
    )
