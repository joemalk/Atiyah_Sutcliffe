import numpy as np

from euclidean_configurations import EuclideanConfiguration, EuclideanConfigurationSet

from time import perf_counter

print("Test 1: Compute AS_det for random configurations with different n's.")
T1_t0 = perf_counter()
for n in range(3,10):
    conf = np.random.random((n,3))
    euc_conf = EuclideanConfiguration(conf)
    print(f'n = {n}: {euc_conf.AS_det()}')
T1_t1 = perf_counter()
print(f'Test 1 completed in {T1_t1-T1_t0} seconds.')

N = 10000
print(f'\nTest 2: compute AS_dets for {N} random configurations with n = 4, \
then find the minimal real part.')
T2_t0 = perf_counter()
confs = np.random.random((N,4,3))
euc_confs = EuclideanConfigurationSet(confs)
print(f'The minimal real part of this sample is: {min(np.real(euc_confs.AS_dets()))}')
T2_t1 = perf_counter()
print(f'Test 2 completed in {T2_t1-T2_t0} seconds.')
