"""DNN (deep neural network) classifier utilities for pp→bbχχ.

Scripts in this package are meant to be run after producing `ppbbchichi-trees.root`.
The training/apply entrypoints are:
- `dnn.train_classifier`: trains a PyTorch MLP on tabular features.
- `dnn.apply_classifier`: applies the trained model and writes `ml_score` back to ROOT.
"""