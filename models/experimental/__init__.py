"""Experimental surrogates not yet wired into the Optimizer dispatch.

These models require different integration patterns (meta-learning loops,
molecular graph inputs) that don't fit the standard suggest() API. They are
provided as standalone implementations for research exploration.

Modules
-------
maml_surrogate
    Model-Agnostic Meta-Learning for few-shot BO across studies.
    Reference: Finn et al. (2017), ICML.

fsbo
    Paper-exact few-shot Bayesian optimization with a meta-trained
    deep-kernel GP, task-scale augmentation, target-task fine-tuning,
    and EI-based search. Reference: Wistuba & Grabocka (2021), ICLR.

warm_start_gp_transfer_baseline
    Approximate transfer-learning baseline that warm-starts GP
    hyperparameters from source-task data, then runs greedy posterior-
    mean search on the target task.

mpnn
    Directed Message Passing Neural Network (D-MPNN) for molecular
    property prediction. Reference: Yang et al. (2019), JCIM.

gps_mpnn
    GPS-style D-MPNN with RWSE positional encodings, global
    self-attention, and cross-component attention.

featurize
    Molecular graph featurization utilities for MPNN inputs.
"""
