"""Main Risk Premium PCA Class"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import qr


class RiskPremiumPCA:
    def __init__(self, gamma, n_factors):
        self._gamma = gamma
        self._k = n_factors
        self.excess_returns = None
        self.loadings = None
        self.factors = None
        self.eigenvalues = None
        self.factor_weights = None
        self.sdf = None
        self.sdf_weights = None
        self.beta = None
        self.alpha = None

    def fit(self, excess_returns, std_norm=False, var_norm=False, orthogonalize=False):
        self.excess_returns = excess_returns.copy()
        X = self.excess_returns.values
        T, N = X.shape
        _1 = np.ones((T, T))
        I = np.eye(T)

        if std_norm:
            # Variance normalization matrix
            V = (1 / T) * X.T @ (I - _1 / T) @ X
            D = np.diag(np.sqrt(np.diag(V)))  # Diag std dev
            WN = np.linalg.inv(D)
        else:
            WN = np.eye(N)

        WT = I + (1 / T) * self._gamma * _1

        # Generic estimator for general weighting matrices
        X_tilde = X @ WN

        # Covariance matrix with weighted mean & truncated decomposition
        V_tilde = (1 / T) * X_tilde.T @ WT @ X_tilde

        # TODO: change the below for a regular eigendecomposition
        trnsf = TruncatedSVD(n_components=self._k)
        trnsf.fit(V_tilde)
        Vh = trnsf.components_
        S = np.diag(trnsf.singular_values_)

        # Lambda_hat are the eigenvectors after reverting the cross-sectional transformation
        Lambda_hat = np.linalg.inv(WN) @ Vh.T

        # Flips the signs of the eigenvectors
        _sign = np.diag(
            np.sign(
                np.mean(
                    X @ Lambda_hat @ np.linalg.inv(Lambda_hat.T @ Lambda_hat), axis=0
                )
            )
        )
        Lambda_hat = Lambda_hat @ _sign

        # Constructing the latent factors
        factorweight = Lambda_hat @ np.linalg.inv(Lambda_hat.T @ Lambda_hat)
        factorweight = factorweight @ np.linalg.inv(
            np.diag(np.sqrt(np.diag(factorweight.T @ factorweight)))
        )
        F_hat = X @ factorweight

        # If variance normalization is True, then the loading are scaled by the eigenvalues. Otherwise they have unit length.
        if var_norm and not orthogonalize:
            Lambda_hat = Lambda_hat @ np.sqrt(S)
            factorweight = factorweight @ np.linalg.inv(np.sqrt(S))
            F_hat = X @ factorweight
            # Here the loadings are normalized to have unit length

        if var_norm and orthogonalize:
            _, R = qr((I - (1 / T) * _1) @ F_hat * (1 / np.sqrt(T)))
            Rotation = np.linalg.inv(R[: self._k, : self._k])
            factorweight = factorweight @ Rotation
            F_hat = X @ factorweight
            signnormalization = np.diag(np.sign(np.mean(F_hat, axis=0)))
            F_hat = F_hat @ signnormalization
            factorweight = factorweight @ signnormalization
            Lambda_hat = Lambda_hat @ np.linalg.inv(Rotation) @ signnormalization

        if not var_norm and orthogonalize:
            _, R = qr((I - (1 / T) * _1) @ F_hat * (1 / np.sqrt(T)))
            Rotation = np.linalg.inv(R[: self._k, : self._k]) @ np.diag(
                np.diag(R[: self._k, : self._k])
            )
            factorweight = factorweight @ Rotation
            F_hat = X @ factorweight
            signnormalization = np.diag(np.sign(np.mean(F_hat, axis=0)))
            F_hat = F_hat @ signnormalization
            factorweight = factorweight @ signnormalization
            Lambda_hat = Lambda_hat @ np.linalg.inv(Rotation) @ signnormalization

        const = {}
        beta = {}
        resid = {}
        sdfweights = {}

        for k in range(self._k):
            factors = F_hat[:, : k + 1]
            # Mean variance optimization

            SDFweights = (
                np.linalg.inv(np.matrix(np.cov(factors.T, rowvar=True)))
                @ np.mean(factors, axis=0).T
            )
            SDF = factors @ SDFweights.T
            SDFweightsassets = (
                Lambda_hat[:, : k + 1]
                @ np.linalg.inv(Lambda_hat[:, : k + 1].T @ Lambda_hat[:, : k + 1])
                @ SDFweights.T
            )

            # Time series regression
            M = np.append(np.ones((T, 1)), factors, axis=1)  # Design  matrix
            dummy = np.linalg.inv(M.T @ M) @ M.T @ X
            residual = X - M @ dummy

            const[k] = dummy[0]
            beta[k] = dummy[1:]
            resid[k] = residual
            sdfweights[k] = SDFweightsassets

        # Processing and saving results
        factor_idx = pd.Index(range(self._k), name="LatentFactor")
        sec_idx = self.excess_returns.columns
        dts_idx = self.excess_returns.index
        res_factorweight = pd.DataFrame(factorweight, columns=factor_idx, index=sec_idx)
        res_loadings = pd.DataFrame(Lambda_hat, columns=factor_idx, index=sec_idx)
        res_factors = pd.DataFrame(F_hat, index=dts_idx, columns=factor_idx)
        res_eig = pd.Series(np.diag(S), index=factor_idx)
        res_sdf = pd.DataFrame(SDF, index=dts_idx).squeeze()
        res_sdfwt = pd.concat([pd.DataFrame(y) for x, y in sdfweights.items()], axis=1)
        res_sdfwt.index = sec_idx
        res_sdfwt.columns = factor_idx
        res_beta = {
            x: pd.DataFrame(y.T, index=sec_idx, columns=factor_idx[: x + 1])
            for x, y in beta.items()
        }
        res_alpha = {x: pd.Series(y.flatten(), index=sec_idx) for x, y in const.items()}

        self.factor_weights = res_factorweight
        self.loadings = res_loadings
        self.factors = res_factors
        self.eigenvalues = res_eig
        self.sdf = res_sdf
        self.sdf_weights = res_sdfwt
        self.beta = res_beta
        self.alpha = res_alpha


if __name__ == "__main__":
    import pandas as pd
    import os

    crsp = pd.read_pickle(f"{os.getcwd()}/data/crsp.pkl")
    rt = crsp["excess_ret"].dropna()

    sample = pd.pivot_table(
        rt.to_frame("rt"), index="date", columns="permno", values="rt"
    )
    sample_ids = sample.count().sort_values().tail(100).index
    sample = sample[sample_ids]

    mdl = RiskPremiumPCA(gamma=0.5, n_factors=4)
    mdl.fit(
        excess_returns=sample,
        std_norm=True,
        orthogonalize=True,
        var_norm=True,
    )
