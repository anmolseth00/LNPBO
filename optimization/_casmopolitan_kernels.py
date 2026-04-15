"""CASMOPOLITAN kernel definitions."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern


class ExponentiatedCategoricalKernel(Kernel):
    """Exponentiated categorical kernel from Wan et al. (2021), Eq. (1)."""

    def __init__(self, n_cat_dims=1, lengthscales=None):
        self.n_cat_dims = n_cat_dims
        if lengthscales is None:
            lengthscales = tuple(np.ones(n_cat_dims, dtype=float).tolist())
        if np.asarray(lengthscales, dtype=float).shape != (n_cat_dims,):
            raise ValueError(f"lengthscales must have shape ({n_cat_dims},)")
        self.lengthscales = lengthscales

    @property
    def hyperparameter_lengthscales(self):
        from sklearn.gaussian_process.kernels import Hyperparameter

        return Hyperparameter("lengthscales", "numeric", (1e-5, 1e3), self.n_cat_dims)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        X_cat = X[:, : self.n_cat_dims]
        Y_cat = Y[:, : self.n_cat_dims]
        match = (X_cat[:, np.newaxis, :] == Y_cat[np.newaxis, :, :]).astype(float)
        ls = np.asarray(self.lengthscales, dtype=float)
        weighted_match = match * ls[np.newaxis, np.newaxis, :]
        exponent = weighted_match.sum(axis=2) / max(self.n_cat_dims, 1)
        K = np.exp(exponent)

        if eval_gradient:
            dK = K[:, :, np.newaxis] * (
                match * ls[np.newaxis, np.newaxis, :] / max(self.n_cat_dims, 1)
            )
            return K, dK
        return K

    def diag(self, X):
        X = np.atleast_2d(X)
        exponent = np.sum(np.asarray(self.lengthscales, dtype=float)) / max(self.n_cat_dims, 1)
        return np.full(X.shape[0], np.exp(exponent))

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        return {"n_cat_dims": self.n_cat_dims, "lengthscales": self.lengthscales}

    def __repr__(self):
        return f"ExponentiatedCategoricalKernel(n_cat={self.n_cat_dims}, lengthscales={self.lengthscales})"

    def clone_with_theta(self, theta):
        clone = ExponentiatedCategoricalKernel(n_cat_dims=self.n_cat_dims, lengthscales=self.lengthscales)
        clone.theta = theta
        return clone

    @property
    def theta(self):
        return np.log(np.asarray(self.lengthscales, dtype=float))

    @theta.setter
    def theta(self, value):
        self.lengthscales = tuple(np.exp(np.asarray(value, dtype=float)).tolist())

    @property
    def bounds(self):
        return np.log(np.tile(np.array([[1e-5, 1e3]], dtype=float), (self.n_cat_dims, 1)))

    @property
    def n_dims(self):
        return self.n_cat_dims


class MixedCasmopolitanKernel(Kernel):
    """Mixed kernel from Wan et al. (2021), Eq. (4)."""

    def __init__(self, n_cat_dims, n_cont_dims, mix_weight=0.5, cat_lengthscales=None):
        self.n_cat_dims = n_cat_dims
        self.n_cont_dims = n_cont_dims
        if cat_lengthscales is None:
            cat_lengthscales = tuple(np.ones(n_cat_dims, dtype=float).tolist())
        if np.asarray(cat_lengthscales, dtype=float).shape != (n_cat_dims,):
            raise ValueError(f"cat_lengthscales must have shape ({n_cat_dims},)")
        self.cat_lengthscales = cat_lengthscales
        self.cat_kernel = ExponentiatedCategoricalKernel(n_cat_dims=n_cat_dims, lengthscales=self.cat_lengthscales)
        self.cont_kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(n_cont_dims),
            nu=2.5,
        )
        mix_weight_value = float(mix_weight)
        if not (0.0 < mix_weight_value < 1.0):
            raise ValueError("mix_weight must lie strictly between 0 and 1.")
        self.mix_weight = mix_weight

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X_cat = X[:, : self.n_cat_dims]
        X_cont = X[:, self.n_cat_dims :]

        if Y is None:
            Y_cat = None
            Y_cont = None
        else:
            Y = np.atleast_2d(Y)
            Y_cat = Y[:, : self.n_cat_dims]
            Y_cont = Y[:, self.n_cat_dims :]

        if not eval_gradient:
            K_cat = self.cat_kernel(X_cat, Y_cat)
            K_cont = self.cont_kernel(X_cont, Y_cont)
            return self.mix_weight * (K_cat * K_cont) + (1.0 - self.mix_weight) * (K_cat + K_cont)

        K_cat, dK_cat = self.cat_kernel(X_cat, Y_cat, eval_gradient=True)
        K_cont, dK_cont = self.cont_kernel(X_cont, Y_cont, eval_gradient=True)
        K_prod = K_cat * K_cont
        K_sum = K_cat + K_cont
        K = self.mix_weight * K_prod + (1.0 - self.mix_weight) * K_sum

        n_cat_params = dK_cat.shape[2]
        n_cont_params = dK_cont.shape[2]
        dK = np.zeros((K.shape[0], K.shape[1], 1 + n_cat_params + n_cont_params))

        lam = self.mix_weight
        dlam_dlogit = lam * (1.0 - lam)
        dK[:, :, 0] = (K_prod - K_sum) * dlam_dlogit

        cat_weight = (lam * K_cont + (1.0 - lam))[:, :, np.newaxis]
        dK[:, :, 1 : 1 + n_cat_params] = cat_weight * dK_cat

        cont_weight = (lam * K_cat + (1.0 - lam))[:, :, np.newaxis]
        dK[:, :, 1 + n_cat_params :] = cont_weight * dK_cont
        return K, dK

    def diag(self, X):
        X = np.atleast_2d(X)
        d_cat = self.cat_kernel.diag(X[:, : self.n_cat_dims])
        d_cont = self.cont_kernel.diag(X[:, self.n_cat_dims :])
        return self.mix_weight * (d_cat * d_cont) + (1.0 - self.mix_weight) * (d_cat + d_cont)

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        params = {
            "n_cat_dims": self.n_cat_dims,
            "n_cont_dims": self.n_cont_dims,
            "mix_weight": self.mix_weight,
            "cat_lengthscales": self.cat_lengthscales,
        }
        if deep:
            cat_params = self.cat_kernel.get_params(deep=True)
            params.update({f"cat_kernel__{k}": v for k, v in cat_params.items()})
            cont_params = self.cont_kernel.get_params(deep=True)
            params.update({f"cont_kernel__{k}": v for k, v in cont_params.items()})
        return params

    def __repr__(self):
        return f"MixedCasmopolitanKernel(n_cat={self.n_cat_dims}, n_cont={self.n_cont_dims}, mix={self.mix_weight:.3f})"

    @property
    def theta(self):
        mix_logit = np.log(self.mix_weight / (1.0 - self.mix_weight))
        return np.concatenate([[mix_logit], self.cat_kernel.theta, self.cont_kernel.theta])

    @theta.setter
    def theta(self, value):
        value = np.asarray(value, dtype=float)
        self.mix_weight = 1.0 / (1.0 + np.exp(-value[0]))
        n_cat = self.cat_kernel.n_dims
        self.cat_kernel.theta = value[1 : 1 + n_cat]
        self.cat_lengthscales = self.cat_kernel.lengthscales
        self.cont_kernel.theta = value[1 + n_cat :]

    @property
    def bounds(self):
        mix_bounds = np.array([[-6.0, 6.0]])
        return np.vstack([mix_bounds, self.cat_kernel.bounds, self.cont_kernel.bounds])

    @property
    def n_dims(self):
        return 1 + self.cat_kernel.n_dims + self.cont_kernel.n_dims


class AdditiveProductKernel(Kernel):
    """Additive decomposition of categorical and continuous kernels."""

    def __init__(self, n_cat_dims, n_cont_dims, lambd=0.5, alpha=1.0, beta=1.0, gamma=1.0):
        self.n_cat_dims = n_cat_dims
        self.n_cont_dims = n_cont_dims
        approx_lengthscale = np.full(n_cat_dims, -np.log(np.clip(lambd, 1e-5, 1.0 - 1e-5)))
        self.cat_kernel = ExponentiatedCategoricalKernel(n_cat_dims=n_cat_dims, lengthscales=approx_lengthscale)
        self.cont_kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(n_cont_dims),
            nu=2.5,
        )
        self.log_alpha = np.log(alpha)
        self.log_beta = np.log(beta)
        self.log_gamma = np.log(gamma)

    @property
    def _alpha(self):
        return np.exp(self.log_alpha)

    @property
    def _beta(self):
        return np.exp(self.log_beta)

    @property
    def _gamma(self):
        return np.exp(self.log_gamma)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X_cat = X[:, : self.n_cat_dims]
        X_cont = X[:, self.n_cat_dims :]

        if Y is None:
            Y_cat = None
            Y_cont = None
        else:
            Y = np.atleast_2d(Y)
            Y_cat = Y[:, : self.n_cat_dims]
            Y_cont = Y[:, self.n_cat_dims :]

        K_cat = self.cat_kernel(X_cat, Y_cat)

        if not eval_gradient:
            K_cont = self.cont_kernel(X_cont, Y_cont)
            return self._alpha * K_cat + self._beta * K_cont + self._gamma * K_cat * K_cont

        K_cont, dK_cont = self.cont_kernel(X_cont, Y_cont, eval_gradient=True)
        K = self._alpha * K_cat + self._beta * K_cont + self._gamma * K_cat * K_cont

        n_cont_params = dK_cont.shape[2]
        dK = np.zeros((K.shape[0], K.shape[1], 3 + n_cont_params))
        dK[:, :, 0] = self._alpha * K_cat
        dK[:, :, 1] = self._beta * K_cont
        dK[:, :, 2] = self._gamma * K_cat * K_cont
        weight = (self._beta + self._gamma * K_cat)[:, :, np.newaxis]
        dK[:, :, 3:] = weight * dK_cont
        return K, dK

    def diag(self, X):
        X = np.atleast_2d(X)
        d_cat = self.cat_kernel.diag(X[:, : self.n_cat_dims])
        d_cont = self.cont_kernel.diag(X[:, self.n_cat_dims :])
        return self._alpha * d_cat + self._beta * d_cont + self._gamma * d_cat * d_cont

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        params = {
            "n_cat_dims": self.n_cat_dims,
            "n_cont_dims": self.n_cont_dims,
            "lambd": float(np.exp(-np.mean(self.cat_kernel.lengthscales))),
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
        }
        if deep:
            cont_params = self.cont_kernel.get_params(deep=True)
            params.update({f"cont_kernel__{k}": v for k, v in cont_params.items()})
        return params

    def __repr__(self):
        return (
            f"AdditiveProductKernel(n_cat={self.n_cat_dims}, "
            f"n_cont={self.n_cont_dims}, "
            f"alpha={self._alpha:.3f}, beta={self._beta:.3f}, "
            f"gamma={self._gamma:.3f}, lambd={float(np.exp(-np.mean(self.cat_kernel.lengthscales))):.3f})"
        )

    @property
    def theta(self):
        return np.concatenate(
            [
                [self.log_alpha, self.log_beta, self.log_gamma],
                self.cont_kernel.theta,
            ]
        )

    @theta.setter
    def theta(self, value):
        self.log_alpha = value[0]
        self.log_beta = value[1]
        self.log_gamma = value[2]
        self.cont_kernel.theta = value[3:]

    @property
    def bounds(self):
        weight_bounds = np.array([[-5.0, 5.0]] * 3)
        return np.vstack([weight_bounds, self.cont_kernel.bounds])

    @property
    def n_dims(self):
        return 3 + self.cont_kernel.n_dims
