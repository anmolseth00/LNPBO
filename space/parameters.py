import numpy as np
from numpy.random import RandomState
from bayes_opt.parameter import BayesParameter

from scipy.optimize import brentq

class ComponentParameter(BayesParameter):
    def __init__(self, name: str, bounds, valid_options) -> None:
        super().__init__(name, bounds)
        self.unique_categories = valid_options

    def __repr__(self) -> str:
        return f"ComponentParameter with {len(self.unique_categories)} options"

    @property
    def is_continuous(self):
        return False

    def random_sample(self, n_samples: int, random_state: RandomState):
        idx = random_state.choice(len(self.unique_categories), size=n_samples)
        return self.unique_categories[idx]

    def to_float(self, value) -> np.ndarray:
        return self.unique_categories[value]

    def to_param(self, value):
        return np.argmin(np.mean((self.unique_categories-value)**2, axis=1))

    def kernel_transform(self, value):
        value = np.atleast_2d(value)
        batch, dim = value.shape
        value = value.reshape((batch, 1, dim))
        idx_closest = np.argmin(np.mean((self.unique_categories-value)**2, axis=-1), 1)
        res = self.unique_categories[idx_closest]
        return res

    def to_string(self, value, str_len) -> str:
        s = f"{value:<{str_len}}"
        if len(s) > str_len:
            if "." in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s

    @property
    def dim(self):
        return self.bounds.shape[0]


class DiscreteParameter(BayesParameter):
    def __init__(self, name: str, domain) -> None:
        self.domain = domain
        bounds = np.array([np.min(domain), np.max(domain)])
        super().__init__(name, bounds)

    def __repr__(self) -> str:
        return f"DiscreteParameter({self.domain})"

    @property
    def is_continuous(self):
        return False

    def random_sample(self, n_samples: int, random_state: RandomState):
        return random_state.choice(self.domain, size=n_samples)

    def to_float(self, value) -> np.ndarray:
        return value

    def to_param(self, value):
        idx_closest = np.argmin((self.domain-value)**2)
        return self.domain[idx_closest]

    def kernel_transform(self, value):
        shape = value.shape
        value = np.atleast_2d(value)
        idx_closest = np.argmin((self.domain-value)**2, 1)
        res = self.domain[idx_closest]
        return res.reshape(shape)

    def to_string(self, value, str_len) -> str:
        s = f"{value:<{str_len}}"
        if len(s) > str_len:
            if "." in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s
    
    @property
    def dim(self):
        return 1


def l1norm(x, axis=-1):
    return x / np.expand_dims(np.sum(x, axis=axis), axis=axis)


class MixtureRatiosParameter(BayesParameter):
    def __init__(self, name: str, nr_components, bounds=None, sum_to=1.) -> None:
        self.domain = range(nr_components)
        self.sum_to = sum_to
        if bounds is None:
            bounds = np.zeros((len(self.domain), 2))
            bounds[:, 1] = 1
        else:
            bounds = np.array(bounds)
        super().__init__(name, bounds)

    def __repr__(self) -> str:
        return f"MixtureRatiosParameter with {len(self.domain)} components"#)"

    @property
    def is_continuous(self):
        # technically this is continuous, but we treat it as non-continuous to trigger DE instead of L-BFGS-B
        return False

    def random_sample(self, n_samples: int, random_state: RandomState):
        res = []
        while len(res) < n_samples:
            candidates = random_state.dirichlet(np.ones(len(self.domain))) * self.sum_to
            if np.all(candidates >= self.bounds[:, 0]) and np.all(candidates <= self.bounds[:, 1]):
                res.append(candidates)
        return np.array(res[:n_samples])
    
    def to_float(self, value) -> np.ndarray:
        return value

    def to_param(self, value):
        return self._project_onto_bounded_simplex(
            value, 
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            self.sum_to
        )

    def kernel_transform(self, value): 
        return self._project_onto_bounded_simplex(
            value, 
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            self.sum_to
        )


    def to_string(self, value, str_len):
        """Represent a parameter value as a string.

        Parameters
        ----------
        value : Any
            The value to represent.

        str_len : int
            The maximum length of the string representation.

        Returns
        -------
        str
        """
        len_each = (str_len - 2) // 3
        str_ = '|'.join([f"{float(np.round(value[i], 4))}"[:len_each] for i in range(3)])
        return str_.ljust(str_len)

    @property
    def dim(self):
        return len(self.domain)

    def _project_onto_bounded_simplex(self, x, l, u, target_sum=1.0):
        """
        Project x onto {y : sum(y) = target_sum, l <= y <= u}
        
        Parameters:
        -----------
        x:
            input array (n,) or (..., n)
        l:
            lower bounds (n,)
        u:
            upper bounds (n,)
        target_sum:
            desired sum (default 1.0)

        """
        original_shape = x.shape
        
        # single vector?
        if x.ndim == 1:
            if (np.all((x >= l) & (x <= u)) and 
                np.abs(np.sum(x) - target_sum) < 1e-10):
                return x
            return self._project_single(x, l, u, target_sum)

        x_flat = x.reshape(-1, x.shape[-1])
        
        # feasibility check
        within_bounds = np.all((x_flat >= l) & (x_flat <= u), axis=1)
        sums = np.sum(x_flat, axis=1)
        sum_satisfied = np.abs(sums - target_sum) < 1e-10
        already_feasible = within_bounds & sum_satisfied

        if np.all(already_feasible):
            return x

        result = x_flat.copy()

        # Only project infeasible vectors
        infeasible_mask = ~already_feasible
        infeasible_indices = np.where(infeasible_mask)[0]

        for i in infeasible_indices:
            result[i] = self._project_single(x_flat[i], l, u, target_sum)

        return result.reshape(original_shape)

    def _project_single(self, x, l, u, target_sum):
        """Project a single vector onto bounded simplex"""
        
        # solvability check
        if np.sum(l) > target_sum or np.sum(u) < target_sum:
            raise ValueError(f"Infeasible bounds: sum(l)={np.sum(l)}, "
                            f"sum(u)={np.sum(u)}, target={target_sum}")

        y = np.clip(x, l, u)

        # already feasible?
        current_sum = np.sum(y)
        if np.abs(current_sum - target_sum) < 1e-10:
            return y

        def residual(lam):
            proj = np.clip(x - lam, l, u)
            return np.sum(proj) - target_sum

        lam_min = np.min(x - u) - 1.0
        lam_max = np.max(x - l) + 1.0

        res_min = residual(lam_min)
        res_max = residual(lam_max)
        
        if res_min * res_max > 0:
            if res_min > 0:
                lam_min -= 10.0
            else:
                lam_max += 10.0

        lam_opt = brentq(residual, lam_min, lam_max, xtol=1e-10)
        return np.clip(x - lam_opt, l, u)
