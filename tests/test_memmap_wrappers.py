import os
import pickle
import datetime
import numpy as np
import gc
import resource
import psutil
import pytest

from pc_komega_utils.memmap_wrappers import mmap_array

from .fixtures import cachedir

def memory_limit(max_mem):
	"""
	Slightly modified version of the code from
	https://stackoverflow.com/questions/46327566/how-to-have-pytest-place-memory-limits-on-tests#46330985
	
	max_mem is the memory limit in GB.
	"""
	def decorator(f):
		def wrapper(*args, **kwargs):
			max_mem_bytes = int(max_mem*1024**3)
			
			process = psutil.Process(os.getpid())
			prev_limits = resource.getrlimit(resource.RLIMIT_DATA)
			resource.setrlimit(
				resource.RLIMIT_DATA, (
					process.memory_info().data + max_mem_bytes, -1
				)
			)
			result = f(*args, **kwargs)
			resource.setrlimit(resource.RLIMIT_DATA, prev_limits)
			return result
		return wrapper
	return decorator

def test_create_delete(cachedir):
	a = mmap_array(cachedir)
	
	assert os.path.isfile(a.tmpfilename)
	
	tmpfilename = a.tmpfilename
	del a
	assert not os.path.exists(tmpfilename)

def test_create_2(cachedir):
	a = mmap_array(cachedir, shape=(3,4,5), dtype=int)
	
	assert a.shape == (3,4,5)
	assert a.dtype == int

def test_pickle(cachedir):
	src = np.reshape(np.arange(6), (2,3))
	
	a = mmap_array(cachedir, shape=src.shape)
	a[:] = src
	
	tmpfile = cachedir/"tmp.pickle"
	
	with open(tmpfile, 'wb') as f:
		pickle.dump(a, f)
	
	del a
	gc.collect()
	
	with open(tmpfile, 'rb') as f:
		b = pickle.load(f)
	
	os.remove(tmpfile)
	
	assert b.shape == src.shape
	assert np.all(b == src)

@memory_limit(0.24)
def test_memory_limit():
	a = np.arange(3e7)

@pytest.mark.xfail #if this fails, memory_limit is working
@memory_limit(0.24)
def test_memory_limit_2():
	a = np.arange(4e7)

@pytest.mark.xfail #if this fails, memory_limit is working
@memory_limit(0.24)
def test_memory_limit_3():
	a = np.arange(3e7)
	b = np.arange(3e7)

@pytest.mark.xfail #currently fails. See the note below.
@memory_limit(0.4)
def test_pickle_large(cachedir):
	"""
	NOTE: The failure of this test suggests that the entire array is being loaded into memory during pickle.dump(a, f) (this is corroborated by a comment in https://github.com/numpy/numpy/issues/22213#issuecomment-1238255808 ). This makes mmap_array useless for my purposes.
	"""
	
	src = np.arange(3e7)
	
	a = mmap_array(cachedir, shape=src.shape)
	a[:] = src
	
	tmpfile = cachedir/"tmp.pickle"
	
	with open(tmpfile, 'wb') as f:
		pickle.dump(a, f)
	
	del a
	gc.collect()
	
	with open(tmpfile, 'rb') as f:
		b = pickle.load(f)
	
	os.remove(tmpfile)
	
	assert b.shape == src.shape
	assert np.all(b == src)
