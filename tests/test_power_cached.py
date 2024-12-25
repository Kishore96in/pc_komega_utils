import os
import datetime
import pickle
import numpy as np
import pytest
import pathlib

import pc_komega_utils
from pc_komega_utils.power.cached import read_power

ignore_invalid_cache = pytest.mark.filterwarnings("ignore::pc_komega_utils.power.cached.InvalidCacheWarning")

def get_datadir():
	module_loc = os.path.dirname(pc_komega_utils.__file__)
	return os.path.join(module_loc, "tests", "data")

def get_cachedir():
	return pathlib.Path(f"/tmp/test_power_cached-{os.getpid()}-{datetime.datetime.now().isoformat()}")

@ignore_invalid_cache
def test_read():
	datadir = get_datadir()
	cachedir = get_cachedir()
	
	p = read_power(
		datadir = datadir,
		cachedir = cachedir,
		quiet = True,
		)
	
	assert len(p.zpos) == 288
	assert len(p.kx) == 101
	assert len(p.ky) == 1
	assert len(p.t) == 2
	assert p.uz_xy.shape == (2, 288, 1, 101)

@ignore_invalid_cache
def test_pickle():
	datadir = get_datadir()
	cachedir = get_cachedir()
	
	p = read_power(
		datadir = datadir,
		cachedir = cachedir,
		quiet = True,
		)
	
	k = p.keys()
	uz_xy = p.uz_xy[()]
	
	s = pickle.dumps(p)
	del p
	
	p2 = pickle.loads(s)
	assert p2.keys() == k
	assert np.all(p2.uz_xy == uz_xy)

@ignore_invalid_cache
def test_coexist():
	"""
	Check that it is possible to open multiple instances at the same time in the same data directory.
	"""
	datadir = get_datadir()
	cachedir = get_cachedir()
	
	p = read_power(
		datadir = datadir,
		cachedir = cachedir,
		quiet = True,
		)
	
	p2 = read_power(
		datadir = datadir,
		cachedir = cachedir,
		quiet = True,
		)

@ignore_invalid_cache
def test_math():
	datadir = get_datadir()
	cachedir = get_cachedir()
	
	p = read_power(
		datadir = datadir,
		cachedir = cachedir,
		quiet = True,
		)
	
	t = p.t[()]
	
	assert np.all(2*p.t == 2*t)
	assert np.all(p.t*4 == t*4)
	assert np.all(p.t/3 == t/3)
	assert np.all(p.t - 3 == t - 3)
	assert np.all(p.t**2 == t**2)
	
	assert np.all(-p.t == -t)
	assert np.all(abs(p.t) == abs(t))
	assert np.all(abs(-p.t) == abs(-t))
	
	assert np.all(p.t == t)
	assert np.all(p.t > t-1)
