from pc_komega_utils.memmap_wrappers import mmap_array
import os

def test_create_delete():
	a = mmap_array("/tmp")
	
	assert os.path.isfile(a.tmpfilename)
	
	tmpfilename = a.tmpfilename
	del a
	assert not os.path.exists(tmpfilename)
