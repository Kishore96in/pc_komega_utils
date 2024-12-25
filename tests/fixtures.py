import pytest
import os
import pathlib
import shutil
import datetime

import pc_komega_utils

@pytest.fixture
def datadir():
	module_loc = os.path.dirname(pc_komega_utils.__file__)
	return os.path.join(module_loc, "tests", "data")

@pytest.fixture
def cachedir():
	cachedir = pathlib.Path(f"/tmp/test_pcko-{os.getpid()}-{datetime.datetime.now().isoformat()}")
	
	if not os.path.exists(cachedir):
		os.makedirs(cachedir)
	elif not os.path.isdir(cachedir):
		raise ValueError(f"Cache directory '{cachedir}' is not a directory")
	
	yield cachedir
	
	#Clean up
	shutil.rmtree(cachedir)

@pytest.fixture
def datadir_tmp(datadir, cachedir):
	shutil.copytree(datadir, cachedir/"data")
	return cachedir/"data"
