import pickle
import os
import numpy as np
import subprocess

from pc_komega_utils.fit import fit_mode, fit_mode_auto

def get_dataset(name):
	dirname = os.path.dirname(__file__)
	dataloc = os.path.join(dirname, f"fit_data/{name}/inputs.pickle")
	
	if not os.path.exists(dataloc):
		p = subprocess.Popen(
			["python", "make_data.py"],
			cwd = os.path.join(os.path.dirname(dataloc)),
			)
		p.wait()
	
	with open(
		dataloc,
		'rb',
		) as f:
		return pickle.load(f)

def test_fit_1():
	dset = get_dataset("1")
	
	fit = fit_mode(
		**dset,
		poly_order = 1,
		n_lorentz = 1,
		om_guess = [0.5],
		gamma_max = 0.1,
		)
	
	omt = dset['omt_near_target']
	d_omt = omt[1] - omt[0]
	
	_, params_lorentz = fit.unpack_params(fit.popt)
	assert np.shape(params_lorentz) == (1,3)
	assert np.isclose(0.5, params_lorentz[0,1], atol=2*d_omt)
