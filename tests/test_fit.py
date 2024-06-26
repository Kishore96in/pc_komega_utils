import numpy as np
import pytest

from pc_komega_utils.fit import (
	fit_mode,
	fit_mode_auto,
	get_mode_eigenfunction,
	)
from pc_komega_utils.models import (
	ModelBaselineExp,
	ModelLineLorentzian,
	AbstractModelMaker,
	)

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
	
	assert np.isclose(1e-5, params_lorentz[0,0], rtol=1e-1)
	assert np.isclose(0.5, params_lorentz[0,1], atol=2*d_omt)
	assert np.isclose(0.01, params_lorentz[0,2], rtol=1e-1)

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
	
	assert np.isclose(1e-5, params_lorentz[i1,0], rtol=1e-1)
	assert np.isclose(0.4, params_lorentz[i1,1], atol=2*d_omt)
	assert np.isclose(0.01, params_lorentz[i1,2], rtol=1e-1)
	
	assert np.isclose(1e-5, params_lorentz[i2,0], rtol=1e-1)
	assert np.isclose(0.6, params_lorentz[i2,1], atol=2*d_omt)
	assert np.isclose(0.02, params_lorentz[i2,2], rtol=1e-1)

def test_fit_auto_2():
	dset = get_dataset_2()
	
	fit = fit_mode_auto(
		**dset,
		poly_order = 1,
		om_guess = [0.4],
		gamma_max = 0.1,
		)
	
	omt = dset['omt_near_target']
	d_omt = omt[1] - omt[0]
	
	_, params_lorentz = fit.unpack_params(fit.popt)
	assert np.shape(params_lorentz) == (2,3)
	
	i1 = np.argmin(np.abs(params_lorentz[:,1] - 0.4)) #Find the Lorentzian closest to omega=0.4
	i2 = int(not i1)
	
	assert np.isclose(1e-5, params_lorentz[i1,0], rtol=1e-1)
	assert np.isclose(0.4, params_lorentz[i1,1], atol=2*d_omt)
	assert np.isclose(0.01, params_lorentz[i1,2], rtol=1e-1)
	
	assert np.isclose(1e-5, params_lorentz[i2,0], rtol=1e-1)
	assert np.isclose(0.6, params_lorentz[i2,1], atol=2*d_omt)
	assert np.isclose(0.02, params_lorentz[i2,2], rtol=1e-1)

def get_dataset_3():
	om = np.linspace(0.3,0.7,100)
	
	A = 1e-11
	om_0 = 0.5
	gam = 0.01
	
	
	data = 3e-10 + om*2e-10 + 1e-10*om**2 + (A*gam/np.pi)/((om - om_0)**2 + gam**2)
	
	return {
		'data_near_target': data,
		'omt_near_target': om,
		}

def test_fit_3():
	dset = get_dataset_3()
	
	fit = fit_mode(
		**dset,
		poly_order = 2,
		n_lorentz = 1,
		om_guess = [0.5],
		gamma_max = 0.1,
		)
	
	omt = dset['omt_near_target']
	d_omt = omt[1] - omt[0]
	
	params_poly, params_lorentz = fit.unpack_params(fit.popt)
	assert np.shape(params_poly) == (3,)
	assert np.shape(params_lorentz) == (1,3)
	
	print(f"{params_poly = }")
	assert np.isclose(3e-10, params_poly[0], rtol=1e-1, atol=0)
	assert np.isclose(2e-10, params_poly[1], rtol=1e-1, atol=0)
	assert np.isclose(1e-10, params_poly[2], rtol=1e-1, atol=0)
	
	assert np.isclose(1e-11, params_lorentz[0,0], rtol=1e-1, atol=0)
	assert np.isclose(0.5, params_lorentz[0,1], atol=2*d_omt)
	assert np.isclose(0.01, params_lorentz[0,2], rtol=1e-1)

def get_dataset_4():
	om = np.linspace(0.2,0.8,100)
	
	A = 1e-5
	om_0 = 0.4
	gam = 0.01
	sigma = 1e-5
	
	A2 = 1e-5
	om_02 = 0.6
	gam2 = 0.02
	
	a = 1e-4
	b = 1
	
	rand = np.random.default_rng(seed=42).normal(loc=0, scale=sigma, size=len(om))
	
	data = (
		+ rand
		+ a*np.exp(-b*om)
		+ (A*gam/np.pi)/((om - om_0)**2 + gam**2)
		+ (A2*gam2/np.pi)/((om - om_02)**2 + gam2**2)
		)
	
	return {
		'data_near_target': data,
		'omt_near_target': om,
		'sigma': sigma,
		}

class ModelMakerLorentzianWithExp(ModelBaselineExp, ModelLineLorentzian, AbstractModelMaker): pass

def test_fit_4():
	dset = get_dataset_4()
	
	fit = fit_mode(
		**dset,
		poly_order = 1,
		n_lorentz = 2,
		om_guess = [0.4],
		gamma_max = 0.1,
		ModelMaker = ModelMakerLorentzianWithExp,
		)
	
	omt = dset['omt_near_target']
	d_omt = omt[1] - omt[0]
	
	params_poly, params_lorentz = fit.unpack_params(fit.popt)
	assert np.shape(params_poly) == (2,)
	assert np.shape(params_lorentz) == (2,3)
	
	i1 = np.argmin(np.abs(params_lorentz[:,1] - 0.4)) #Find the Lorentzian closest to omega=0.4
	i2 = int(not i1)
	
	assert np.isclose(1e-4, params_poly[0], rtol=1e-1)
	assert np.isclose(1, params_poly[1], rtol=1e-1)
	
	assert np.isclose(1e-5, params_lorentz[i1,0], rtol=1e-1)
	assert np.isclose(0.4, params_lorentz[i1,1], atol=2*d_omt)
	assert np.isclose(0.01, params_lorentz[i1,2], rtol=1e-1)
	
	assert np.isclose(1e-5, params_lorentz[i2,0], rtol=1e-1)
	assert np.isclose(0.6, params_lorentz[i2,1], atol=2*d_omt)
	assert np.isclose(0.02, params_lorentz[i2,2], rtol=1e-1)

def test_get_mode_eigenfunction_1():
	dset = get_dataset_1()
	
	dummy_getter = lambda *args: dset
	mass, = get_mode_eigenfunction(
		omega_0=0.5,
		om_tilde_min=0.3,
		om_tilde_max=0.7,
		poly_order = 1,
		#
		dr=None,
		k_tilde=None,
		z_list=[1],
		omega_tol = 0.05,
		mode_mass_method="sum",
		getter = dummy_getter,
		gamma_max=0.1
		)
	
	assert np.isclose(mass, 2.4e-3, rtol=1e-2)

def test_get_mode_eigenfunction_1_err():
	dset = get_dataset_1()
	
	dummy_getter = lambda *args: dset
	res = get_mode_eigenfunction(
		omega_0=0.5,
		om_tilde_min=0.3,
		om_tilde_max=0.7,
		poly_order = 1,
		#
		dr=None,
		k_tilde=None,
		z_list=[1],
		omega_tol = 0.05,
		mode_mass_method="sum",
		getter = dummy_getter,
		gamma_max=0.1,
		full_output = True,
		)
	
	#Values below are directly copied from the output; have not independently checked them.
	mass_list = res.mass
	err_list = res.error
	assert len(mass_list) == 1
	assert len(err_list) == 1
	assert np.isclose(mass_list[0], 2.4e-3, rtol=1e-2)
	assert np.isclose(err_list[0], 4.9e-5, rtol=1e-2)
