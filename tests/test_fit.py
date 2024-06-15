import numpy as np
import subprocess

from pc_komega_utils.fit import fit_mode, fit_mode_auto

def get_dataset_1():
	om = np.linspace(0.3,0.7,100)
	
	A = 1e-5
	om_0 = 0.5
	gam = 0.01
	sigma = 1e-5
	
	rand = np.random.default_rng(seed=42).normal(loc=1e-4, scale=sigma, size=len(om))
	
	data = rand + (A*gam/np.pi)/((om - om_0)**2 + gam**2)
	
	return {
		'data_near_target': data,
		'omt_near_target': om,
		'sigma': sigma,
		}

def test_fit_1():
	dset = get_dataset_1()
	
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

def get_dataset_2():
	om = np.linspace(0.2,0.8,100)
	
	A = 1e-5
	om_0 = 0.4
	gam = 0.01
	sigma = 1e-5
	
	A2 = 1e-5
	om_02 = 0.6
	gam2 = 0.02
	
	rand = np.random.default_rng(seed=42).normal(loc=1e-4, scale=sigma, size=len(om))
	
	data = (
		+ rand
		+ om*1e-4
		+ (A*gam/np.pi)/((om - om_0)**2 + gam**2)
		+ (A2*gam2/np.pi)/((om - om_02)**2 + gam2**2)
		)
	
	return {
		'data_near_target': data,
		'omt_near_target': om,
		'sigma': sigma,
		}

def test_fit_2():
	dset = get_dataset_2()
	
	fit = fit_mode(
		**dset,
		poly_order = 1,
		n_lorentz = 2,
		om_guess = [0.4],
		gamma_max = 0.1,
		)
	
	omt = dset['omt_near_target']
	d_omt = omt[1] - omt[0]
	
	_, params_lorentz = fit.unpack_params(fit.popt)
	assert np.shape(params_lorentz) == (2,3)
	
	i1 = np.argmin(np.abs(params_lorentz[:,1] - 0.4)) #Find the Lorentzian closest to omega=0.4
	i2 = int(not i1)
	assert np.isclose(0.4, params_lorentz[i1,1], atol=2*d_omt)
	assert np.isclose(0.6, params_lorentz[i2,1], atol=2*d_omt)

def test_fit_auto_2():
	dset = get_dataset_2()
	
	fit = fit_mode_auto(
		**dset,
		poly_order = 1,
		om_guess = [0.4],
		gamma_max = 0.1,
		threshold_p = 0.5,
		)
	
	omt = dset['omt_near_target']
	d_omt = omt[1] - omt[0]
	
	_, params_lorentz = fit.unpack_params(fit.popt)
	assert np.shape(params_lorentz) == (2,3)
	
	i1 = np.argmin(np.abs(params_lorentz[:,1] - 0.4)) #Find the Lorentzian closest to omega=0.4
	i2 = int(not i1)
	assert np.isclose(0.4, params_lorentz[i1,1], atol=2*d_omt)
	assert np.isclose(0.6, params_lorentz[i2,1], atol=2*d_omt)
