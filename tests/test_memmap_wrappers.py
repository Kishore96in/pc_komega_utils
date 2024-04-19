import os

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
