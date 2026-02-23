from __future__ import annotations
from bayes_opt import acquisition
from bayes_opt.target_space import TargetSpace
from copy import deepcopy
import numpy as np

    
class KrigingBeliever(acquisition.AcquisitionFunction):
    def __init__(self, base_acquisition: acquisition.AcquisitionFunction, random_state=None, atol=1e-5, rtol=1e-8) -> None:
        super().__init__(random_state)
        self.base_acquisition = base_acquisition
        self.dummies = []
        self.atol = atol
        self.rtol = rtol

    def base_acq(self, *args, **kwargs):
        return self.base_acquisition.base_acq(*args, **kwargs)
    
    def clear_dummies(self):
        self.dummies = []

    def _ensure_dummies_match(self) -> None:
        if self.dummy_constraints:
            if len(self.dummy_constraints) != len(self.dummy_targets):
                msg = (
                    "Number of dummy constraints " +
                    f"{len(self.dummy_constraints)} doesn't match number of " +
                    f" dummy targets {len(self.dummy_targets)}. This can " +
                    "happen if constrained and unconstrained optimization is "+
                    "mixed."
                )
                raise ValueError(msg)

    def _remove_expired_dummies(self, target_space: TargetSpace) -> None:
        dummies = []
        for dummy in self.dummies:
            close = np.isclose(dummy, target_space.params, rtol=self.rtol, atol=self.atol)
            if not close.all(axis=1).any():
                dummies.append(dummy)
        self.dummies = dummies
    
    def _create_dummy_target_space(self, gp, target_space: TargetSpace, fit_gp: bool=True) -> TargetSpace:
        # Check if any dummies have been evaluated and remove them
        self._remove_expired_dummies(target_space)
        if fit_gp:
            self._fit_gp(gp, target_space)
        # Create a copy of the target space
        dummy_target_space = deepcopy(target_space)

        if self.dummies:
            dummy_targets = gp.predict(np.array(self.dummies).reshape((len(self.dummies), -1)))
            if dummy_target_space.constraint is not None:
                dummy_constraints = target_space.constraint.approx(np.array(self.dummies).reshape((len(self.dummies), -1)))
            for idx, dummy in enumerate(self.dummies):
                if dummy_target_space.constraint is not None:
                    dummy_target_space.register(dummy, dummy_targets[idx].squeeze(), dummy_constraints[idx].squeeze())
                else:
                    dummy_target_space.register(dummy, dummy_targets[idx].squeeze())
        return dummy_target_space

    def suggest(self, gp, target_space: TargetSpace, n_random=10_000, n_smart=10, fit_gp:bool=True, random_state=None) -> np.ndarray:
        if len(target_space) == 0:
            raise ValueError("Cannot suggest a point without previous samples. Use target_space.random_sample() to generate a point.")

        # fit GP only if necessary
        # GP needs to be fitted to predict dummy targets
        dummy_target_space = self._create_dummy_target_space(gp, target_space, fit_gp=fit_gp)

        # Create a copy of the GP
        dummy_gp = deepcopy(gp)
        # Always fit dummy GP!
        x_max = self.base_acquisition.suggest(
            dummy_gp,
            dummy_target_space,
            n_random=n_random,
            n_smart=n_smart,
            fit_gp=True,
            random_state=random_state
        )
        self.dummies.append(x_max)

        return x_max
