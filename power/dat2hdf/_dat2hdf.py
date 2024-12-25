"""
Convert power*_xy.dat to a HDF5 file containing the same information
"""

import h5py
import pathlib
import numpy as np
import pencil as pc

def powerxy_to_hdf5(power_name, file_name, datadir, out_file=None):
	"""
	Logic is copied from pc.read.powers.Power._read_power2d
	"""
	
	datadir = pathlib.Path(datadir)
	
	if file_name[-4:] == ".dat":
		basename = file_name[:-4]
	else:
		basename = file_name
	
	if out_file is None:
		out_file = datadir/f"{basename}.h5"
	
	dtype_f = np.single
	dtype_c = np.csingle
	dtype_i = np.int64
	
	dim = pc.read.dim(datadir=datadir)
	param = pc.read.param(datadir=datadir)
	
	with (
		h5py.File(out_file, mode='x') as f_out,
		open(datadir/file_name, mode='r') as f,
		):
		_ = f.readline()  # ignore first line
		header = f.readline()
		
		# Get k vectors:
		if param.lintegrate_shell:
			nk = int(
				header
				.split()[header.split().index("k") + 1]
				.split(")")[0][1:]
				)
			k = []
			for _ in range(int(np.ceil(nk / 8))):
				line = f.readline()
				k.extend([float(j) for j in line.split()])
			k = np.array(k, dtype=dtype_f)
			f_out.create_dataset("k", data=k)
		else:
			nkx = int(
				header
				.split()[header.split().index("k_x") + 1]
				.split(")")[0][1:]
				)
			kx = []
			for _ in range(int(np.ceil(nkx / 8))):
				line = f.readline()
				kx.extend([float(j) for j in line.split()])
			kx = np.array(kx, dtype=dtype_f)
			f_out.create_dataset("kx", data=kx)

			nky = int(
				header
				.split()[header.split().index("k_y") + 1]
				.split(")")[0][1:]
				)
			ky = []
			for _ in range(int(np.ceil(nky / 8))):
				line = f.readline()
				ky.extend([float(j) for j in line.split()])
			ky = np.array(ky, dtype=dtype_f)
			f_out.create_dataset("ky", data=ky)

			nk = nkx * nky

		# Now read z-positions, if any
		if param.lintegrate_z:
			nzpos = 1
		else:
			ini = f.tell()
			line = f.readline()
			if "z-pos" in line:
				nzpos = int(re.search(r"\((\d+)\)", line)[1])
				block_size = int(np.ceil(nzpos / 8))
				zpos = []
				for _ in range(block_size):
					line = f.readline()
					zpos.extend([float(j) for j in line.split()])
				f_out.create_dataset("zpos", data=np.array(zpos, dtype=dtype_f))
			else:
				# there was no list of z-positions, so reset the position of the reader.
				f.seek(ini)

				nzpos = dim.nzgrid
				grid = pc.read.grid(datadir=datadir, trim=True, quiet=True)
				f_out.create_dataset("zpos", data=grid.z, dtype=dtype_f)

		# Now read the rest of the file
		time = []
		its = []

		if param.lintegrate_shell:
			block_size = np.ceil(nk / 8) * nzpos + 1
		else:
			block_size = np.ceil(int(nk * nzpos) / 8) + 1
		
		f_out.create_group(power_name)
		f_out.create_dataset("version", data=np.array([1,0,0], dtype=dtype_i))
		
		
		def _write_power_array(power_array, it):
			if param.lintegrate_shell or (dim.nxgrid == 1 or dim.nygrid == 1):
				power_array = power_array.reshape([nzpos, nk])
			else:
				power_array = power_array.reshape([nzpos, nky, nkx])
			
			f_out[power_name].create_dataset(str(it), data=power_array)
		
		first = True
		it = 0
		for line_idx, line in enumerate(f):
			if line_idx % block_size == 0:
				if not first:
					_write_power_array(power_array, it)
				
				first = False
				
				t = float(line.strip())
				it += 1
				time.append(t)
				its.append(it)
				
				if param.lcomplex:
					power_array = np.zeros(nzpos*nk, dtype=dtype_c)
				else:
					power_array = np.zeros(nzpos*nk, dtype=dtype_f)
				
				ik = 0
			else:
				lsp = line.strip().split()

				if param.lcomplex:
					# complex power spectrum
					real = lsp[0::2]
					imag = lsp[1::2]
					for a, b in zip(real, imag):
						power_array[ik] = float(a) + 1j * float(b)
						ik += 1
				else:
					for value_string in lsp:
						power_array[ik] = float(value_string)
						ik += 1
		
		#Last one has still not been written
		_write_power_array(power_array, it)
		
		f_out.create_group("times")
		f_out['times'].create_dataset("it", data=np.array(its, dtype=dtype_i))
		f_out['times'].create_dataset("t", data=np.array(time, dtype=dtype_f))
		
		f_out.create_dataset("nzpos", data=nzpos, dtype=dtype_i)
