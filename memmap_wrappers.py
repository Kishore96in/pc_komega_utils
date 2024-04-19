"""
Helpers to handle arrays and lists larger than RAM.
"""

import numpy as np
import os
import datetime
import sys

class mmap_array(np.memmap):
	def __new__(cls, cache_location=None, **kwargs):
		if cache_location is None:
			raise ValueError
		
		if not os.path.exists(cache_location):
			os.makedirs(cache_location)
		elif not os.path.isdir(cache_location):
			raise ValueError
		
		if 'dtype' not in kwargs:
			kwargs['dtype'] = np.double
		
		tmpfilename = os.path.join(cache_location, f"memmap-{os.getpid()}-{datetime.datetime.now().isoformat()}")
		
		with open(tmpfilename, mode='wb') as f:
			#NOTE: this is to circumvent "ValueError: cannot mmap an empty file" in np.memmap.__new__
			dtype_size = sys.getsizeof(kwargs['dtype'](0))
			f.write(dtype_size*b"\0")
		
		obj = super().__new__(cls, filename=tmpfilename, **kwargs)
		obj.tmpfilename = tmpfilename
		
		return obj
		
	def __del__(self):
		if hasattr(self, 'tmpfilename'):
			os.remove(self.tmpfilename)
