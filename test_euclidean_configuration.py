import numpy as np

from euclidean_configuration import EuclideanConfiguration

print("Compute AS_det for random configurations with different n's")
for n in range(3,10):
    conf = np.random.random((n,3))
    euc_conf = EuclideanConfiguration(conf)
    print(f'n = {n}: {euc_conf.AS_det()}')

