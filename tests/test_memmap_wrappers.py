import os
import pickle
import datetime
import numpy as np
import gc
import resource
import psutil

from pc_komega_utils.memmap_wrappers import mmap_array

def memory_limit(max_mem):
	"""
	https://stackoverflow.com/questions/46327566/how-to-have-pytest-place-memory-limits-on-tests#46330985
	"""
	def decorator(f):
		def wrapper(*args, **kwargs):
			process = psutil.Process(os.getpid())
			prev_limits = resource.getrlimit(resource.RLIMIT_AS)
			resource.setrlimit(
				resource.RLIMIT_AS, (
					process.memory_info().rss + max_mem, -1
				)
			)
			result = f(*args, **kwargs)
			resource.setrlimit(resource.RLIMIT_AS, prev_limits)
			return result
		return wrapper
	return decorator

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
