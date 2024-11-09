"""Main Risk Premium PCA Class"""
import numpy as np
from sklearn.decomposition import TruncatedSVD


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
            V = (1 / T) * X.T @ (I - _1_t / T) @ X
            D = np.diag(np.sqrt(np.diag(V)))  # Diag std dev
            WN = np.linalg.inv(D)
        else:
            WN = np.eye(N)

        WT = I + (1 / T) * self._gamma * _1_t

        # Generic estimator for general weighting matrices
        X_tilde = X @ WN

        # Covariance matrix with weighted mean & truncated decomposition
        V_tilde = (1 / T) * X_tilde.T @ WT @ X_tilde

        trnsf = TruncatedSVD(n_components=self._k)
        trnsf.fit(V_tilde)
        Vh = trnsf.components_
        S = np.diag(trnsf.singular_values_)

        # V_hat are the eigenvectors after reverting the cross-sectional transformation
        V_hat = np.linalg.inv(WN) @ Vh.T

        # Flips the signs of the eigenvectors
        _sign = np.diag(np.sign(np.mean(X @ V_hat @ np.linalg.inv(V_hat.T @ V_hat), axis=0)))
        V_hat = V_hat @ _sign

        # Constructing the latent factors
        factorweight = V_hat @ np.linalg.inv(V_hat.T @ V_hat)
        factorweight = factorweight @ np.linalg.inv(np.diag(np.sqrt(np.diag(factorweight.T @ factorweight))))
        F_hat = X @ factorweight

        # If variance normalization is True, then the loading are scaled by the eigenvalues. Otherwise they have unit length.
        if var_norm and not orthogonalize:
            V_hat
            # Here the loadings are normalized to have unit length



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
