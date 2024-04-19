import os
import pickle
import datetime
import numpy as np
import gc

from pc_komega_utils.memmap_wrappers import mmap_array

def test_create_delete():
	a = mmap_array("/tmp")
	
	assert os.path.isfile(a.tmpfilename)
	
	tmpfilename = a.tmpfilename
	del a
	assert not os.path.exists(tmpfilename)

def test_create_2():
	a = mmap_array("/tmp", shape=(3,4,5), dtype=int)
	
	assert a.shape == (3,4,5)
	assert a.dtype == int

def test_pickle():
	src = np.reshape(np.arange(6), (2,3))
	
	a = mmap_array("/tmp", shape=src.shape)
	a[:] = src
	
	tmpfile = f"/tmp/test_mmap_wrapper-{os.getpid()}-{datetime.datetime.now().isoformat()}"
	
	with open(tmpfile, 'wb') as f:
		pickle.dump(a, f)
	
	del a
	gc.collect()
	
	with open(tmpfile, 'rb') as f:
		b = pickle.load(f)
	
	os.remove(tmpfile)
	
	assert b.shape == src.shape
	assert np.all(b == src)
