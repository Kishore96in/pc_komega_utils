import numpy as np

from pc_komega_utils.utils import smooth_gauss

def test_smooth_gauss_multidim():
	"""
	Check handling of multidimensional arrays in smooth_gauss.
	"""
	data = np.random.default_rng(seed=42).uniform(size=10)
	
	sm_1d = smooth_gauss(data, 1, axis=0)
	
	data_2d = np.stack([data, data, data])
	data_3d = np.stack([data_2d, data_2d])
	
	sm_2d = smooth_gauss(data_2d, 1, axis=-1)
	assert np.all(sm_1d == sm_2d[0,:])
	
	sm_3d = smooth_gauss(data_3d, 1, axis=-1)
	assert np.all(sm_1d == sm_3d[0,0,:])
	
	sm_3ds = smooth_gauss(np.swapaxes(data_3d,1,2), 1, axis=1)
	assert np.all(sm_1d == sm_3ds[0,:,0])

def test_smooth_gauss_multidim2():
	"""
	Check handling of multidimensional arrays in smooth_gauss.
	"""
	data1 = np.random.default_rng(seed=42).uniform(size=10)
	data2 = np.random.default_rng(seed=42).uniform(size=10)
	
	sm1_1d = smooth_gauss(data1, 1)
	sm2_1d = smooth_gauss(data2, 1)
	
	data_2d = np.stack([data1, data2])
	
	sm_2d = smooth_gauss(data_2d, 1, axis=1)
	assert np.all(sm1_1d == sm_2d[0,:])
	assert np.all(sm2_1d == sm_2d[1,:])

