"""Experimental surrogates not yet wired into the Optimizer dispatch.

These models require different integration patterns (meta-learning loops,
molecular graph inputs) that don't fit the standard suggest() API. They are
provided as standalone implementations for research exploration.

Modules
-------
maml_surrogate
    Model-Agnostic Meta-Learning for few-shot BO across studies.
    Reference: Finn et al. (2017), ICML.

fsbo_surrogate
    Warm-started GP hyperparameters from meta-training data.
    Inspired by FSBO (Wistuba & Grabocka, ICLR 2021).

mpnn
    Directed Message Passing Neural Network (D-MPNN) for molecular
    property prediction. Reference: Yang et al. (2019), JCIM.

gps_mpnn
    GPS-style D-MPNN with RWSE positional encodings, global
    self-attention, and cross-component attention.

featurize
    Molecular graph featurization utilities for MPNN inputs.
"""
