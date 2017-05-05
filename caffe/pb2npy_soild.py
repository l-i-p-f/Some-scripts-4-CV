import numpy as np

MEAN_NPY_PATH = 'mean.npy'

mean = np.ones([3,256, 256], dtype=np.float)
mean[0,:,:] = 104
mean[1,:,:] = 117
mean[2,:,:] = 123

np.save(MEAN_NPY_PATH, mean)
