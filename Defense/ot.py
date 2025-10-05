
# from geomloss import SamplesLoss
# import inspect
# print(inspect.signature(SamplesLoss(loss="sinkhorn")))
# print(inspect.signature(SamplesLoss(loss="sinkhorn").forward))
# help(SamplesLoss)



from fidelity_eval import wasserstein_distance, jensen_shannon_distance, maximum_mean_discrepancy
import torch
import numpy as np


X = torch.rand(25000, 8)
Y = torch.rand(25000, 8)

# Compute Wasserstein distance (Sinkhorn divergence)
loss1 = wasserstein_distance(X, Y)
loss2 = jensen_shannon_distance(X, Y)
loss3= maximum_mean_discrepancy(X, Y)

print("OK, wasserstein on balanced:", float(loss1))
print("OK, jensen_shannon on balanced:", float(loss2))
print("OK, maximum_mean_discrepancy on balanced:", float(loss3))


X = torch.rand(25000, 8)
Y = torch.rand(2500, 8)

# Compute Wasserstein distance (Sinkhorn divergence)
loss1 = wasserstein_distance(X, Y)
loss2 = jensen_shannon_distance(X, Y)
loss3= maximum_mean_discrepancy(X, Y)

print("OK, wasserstein on unbalanced:", float(loss1))
print("OK, jensen_shannon on unbalanced:", float(loss2))
print("OK, maximum_mean_discrepancy on unbalanced:", float(loss3))

X = torch.rand(25000, 8)
Y = torch.rand(250, 8)

# Compute Wasserstein distance (Sinkhorn divergence)
loss1 = wasserstein_distance(X, Y)
loss2 = jensen_shannon_distance(X, Y)
loss3= maximum_mean_discrepancy(X, Y)

print("OK, wasserstein on more unbalanced:", float(loss1))
print("OK, jensen_shannon on more unbalanced:", float(loss2))
print("OK, maximum_mean_discrepancy on more unbalanced:", float(loss3))