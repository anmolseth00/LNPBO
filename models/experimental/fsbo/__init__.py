"""Paper-exact FSBO implementation.

This package contains the exact few-shot Bayesian optimization path based on
the deep-kernel Gaussian-process surrogate from Wistuba & Grabocka (2021).
"""

from .adapt import (
    build_cold_gp_model,
    build_meta_initialized_model,
    expected_improvement,
    optimize_gp_model,
    predict_mean_std,
)
from .encoder import FSBOFeatureExtractor, IdentityFeatures, normalize_features, to_tensor
from .evaluate import (
    compute_source_task_loss_matrix,
    evolutionary_warm_start,
    main,
    random_baseline,
    run_fsbo_bo_loop,
)
from .kernel import DeepKernelExactGP, make_fsbo_kernel
from .meta_train import FSBOMetaState, augment_task_labels, meta_train_fsbo, sample_task_scaling_bounds

__all__ = [
    "DeepKernelExactGP",
    "FSBOFeatureExtractor",
    "FSBOMetaState",
    "IdentityFeatures",
    "augment_task_labels",
    "build_cold_gp_model",
    "build_meta_initialized_model",
    "compute_source_task_loss_matrix",
    "evolutionary_warm_start",
    "expected_improvement",
    "main",
    "make_fsbo_kernel",
    "meta_train_fsbo",
    "normalize_features",
    "optimize_gp_model",
    "predict_mean_std",
    "random_baseline",
    "run_fsbo_bo_loop",
    "sample_task_scaling_bounds",
    "to_tensor",
]
