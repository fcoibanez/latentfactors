"""Functions used in 'Exchange rates and macroeconomic fundamentals: Evidence of instabilities from time‐varying factor loadings'"""

import numpy as np


def KFNomean(vP, Data, matF):
    """
    The Kalman filter for DFM by concentrating out means

    Input:    vP - n-by-1 numpy vector of parameters, no mean parameters
              Data - x-by-1 numpy vector of observations
              matF - principal components/factors
    Output:   vF - vector of prediction error variances
              vV - vector of prediction errors for data
              mVf - matrix of prediction errors for factors
              vMean - Mean estimate
    """

    # Specify the matrices
    iR = matF.shape[1]
    f = matF
    T = np.eye(iR)
    Q = np.eye(iR)
    P_t = np.eye(iR)
    for j in range(iR):
        T[j, j] = vP[2 * j].item()
        Q[j, j] = vP[2 * j + 1].item()
        P_t[j, j] = Q[j, j].item() / (1 - T[j, j].item() ** 2)

    sigmaeps = vP[-1].item()

    # KF
    iT = len(Data)  # Now determined by the length of the Data vector
    vF = np.zeros(iT + 1)
    vV = np.zeros(iT + 1)
    mVf = np.zeros((iT + 1, iR))
    a_t = np.zeros(iR)  # Estimate mean in measurement eq
    Af_t = np.zeros((iR, iR))

    for i in range(iT):
        vV[i] = Data[i].item() - f[i, :] @ a_t  # Data prediction error
        mVf[i, :] = f[i, :] - f[i, :] @ Af_t  # Factor prediction error
        vF[i] = f[i, :] @ P_t @ f[i, :].T + sigmaeps
        temp = P_t @ f[i, :].T / vF[i]  # Gain
        a_tt = a_t + temp * vV[i]  # Updating equation
        Af_tt = Af_t + temp.reshape(-1, 1) @ mVf[i, :].reshape(1, -1)
        P_tt = (
            P_t - temp.reshape(-1, 1) @ f[i, :].reshape(1, -1) @ P_t
        )  # Updating equation
        a_t = T @ a_tt  # Prediction equation (Parameters)
        Af_t = T @ Af_tt
        P_t = T @ P_tt @ T.T + Q  # Prediction equation (Parameter variance)

    vV = vV[:iT]  # Delete the last element as it is only relevant for forecasting
    mVf = mVf[:iT, :]  # Delete the last element as it is only relevant for forecasting
    vF = vF[:iT]  # Same here
    XdivF = (mVf / vF[:, np.newaxis]).T
    vMean = np.linalg.solve(XdivF @ mVf, XdivF @ vV)  # GLS estimate

    return vV, mVf, vF, vMean


def LoglikNomean(vP, Data, mF):
    """
    Calculates the loglikelihood function by concentrating out the mean parameters

    Input:    vP - vector of parameters, not including means
              Data - vector of data
              mF - principal components/Factors
    Output:   Value - loglikelihood value
    """

    iR = mF.shape[1]  # Number of factors

    vPtrans = (
        vP.copy().reshape(-1, 1)
    )  # Vector of parameters (create a copy to avoid modifying the original)
    for i in range(iR):
        vPtrans[2 * i + 1] = np.exp(vP[2 * i + 1])

    vPtrans[-1] = np.exp(vP[-1])

    vV, vVf, vF, vMean = KFNomean(vPtrans, Data, mF)

    res = vV - vVf @ vMean
    Value = 0.5 * np.sum(np.log(vF)) + 0.5 * (res / vF) @ res  # Log-Likelihood

    return Value


if __name__ == "__main__":
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from latentfactors import cfg

    k = 3

    fx = pd.read_excel(rf"{cfg.fldr}/data/hmsu/fx.xlsx", index_col=0)
    eco_us = pd.read_excel(rf"{cfg.fldr}/data/hmsu/usa.xlsx", index_col=0)

    # Log-difference FX
    dlog_fx = fx.apply(np.log).diff().reindex(eco_us.index).mul(-1)
    t, n = dlog_fx.shape

    # Factor estimation
    scaler = StandardScaler()
    scaler.fit(eco_us)
    X = scaler.transform(eco_us)

    # Transform them into actors
    np.random.seed(1234)
    svd = TruncatedSVD(n_components=k)
    svd.fit(X)
    V = svd.components_  # Right singular vectors
    S = svd.singular_values_

    U = X @ V.T @ np.linalg.inv(np.diag(S))  # Left singular values
    mF = np.multiply(U, np.sqrt(t))

    mBeta = {}

    # For-loop
    crncy = "AUD"

    # OLS Estimates
    r = dlog_fx[crncy].values.reshape(-1, 1)  # FX Series
    b = np.linalg.inv(mF.T @ mF) @ mF.T @ r  # OLS Beta
    r_hat = mF @ b  # OLS predicted values
    e = r - r_hat
    s_e = e.T @ e / t  # Asymptotic OLS variance
    L_ols = (
        -1 / 2 * t * np.log(s_e.item())
    )  # Likelihood values for restricted model with all loadings constant

    vP0 = np.concatenate([np.zeros((2 * k, 1)), np.log(s_e)])

    ll = LoglikNomean(vP=vP0.copy(), Data=r.copy(), mF=mF.copy())
    print(ll)
