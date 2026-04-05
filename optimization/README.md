# optimization/

Core Bayesian optimization engine. Contains the `Optimizer` class (unified API for all surrogate/acquisition/batch combinations), the BoTorch GP-BO pipeline (`gp_bo.py`), discrete pool scoring for tree-based surrogates (`discrete.py`), acquisition functions (UCB, EI, LogEI, Local Penalization, Kriging Believer, Thompson Sampling, GIBBON), kernel implementations (Matern, Tanimoto, Aitchison, DKL, RF-Kernel, Compositional), and CASMOPolitan mixed-variable optimization.

This is the production engine used by the CLI (`lnpbo suggest`) and the benchmark harness. Research/experimental surrogates live in `models/`.
