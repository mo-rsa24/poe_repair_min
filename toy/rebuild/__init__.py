"""PyTorch rebuild of notebooks/simple_distributions.ipynb, extended into the composition testbed.

Layer 1: datasets.py, model.py, train.py (the notebook's toys, ResNet score net, EBM wrapper, cosine schedule).
Layer 2: compose.py, samplers.py (product, mixture, negation; annealed samplers from anneal_samplers.py).
Layer 3: ideal.py, kinds.py (ideal experts, exact targets, the seven kinds).
Layer 4: parts_world.py (16-pixel Gaussian-mixture experts with exact products).
Layer 5: overlap_dial.py (A-and-B share in each training set, and the fine-tune variant).
"""
