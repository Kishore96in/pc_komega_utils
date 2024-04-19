"""
Helpers to handle arrays and lists larger than RAM.
"""

import numpy as np
import os
import datetime

class mmap_array(np.memmap):
	def __new__(cls, cache_location=None, **kwargs):
		if cache_location is None:
			raise ValueError
		
		if not os.path.exists(cache_location):
			os.makedirs(cache_location)
		elif not os.path.isdir(cache_location):
			raise ValueError
		
		if 'dtype' not in kwargs:
			kwargs['dtype'] = float
		
		tmpfilename = os.path.join(cache_location, f"memmap-{os.getpid()}-{datetime.datetime.now().isoformat()}")
		
		with open(tmpfilename, mode='w') as f:
			#NOTE: this is to circumvent "ValueError: cannot mmap an empty file" in np.memmap.__new__
			f.write("a")
		
		obj = super().__new__(cls, filename=tmpfilename, **kwargs)
		obj.tmpfilename = tmpfilename
		
		return obj
		
	def __del__(self):
		os.remove(self.tmpfilename)
