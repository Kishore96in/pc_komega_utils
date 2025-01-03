# Changelog

## 3.2

### Added
- power.decimate.decimate_power_obj takes an optional argument `keeponly`
- New submodule power.dat2h5

## 3.1

### Changed
- PowerCached: keys from the HDF5 file are now added to `__dict__`

### Added
- model.AbstractModelMaker has a new property, ind_line_freq
- power.decimate.decimate_power_obj

## 3.0

### Changed
- power_cached has been moved to power.cached

### Added
- power.decimate: postprocessing utility to extract power_spectra only at a subset of z values and save them to a HDF5 file.

## 2.0

### Changed
- pcko.getters.AbstractModelMaker.line is now expected to accept arbitrary kwargs.
- Internal calls of AbstractModelMaker.line now pass params_poly as a kwarg (to allow line profiles that depend on some parameter(s) of the continuum)

## 1.1

### Changed
- fit_mode and its wrappers now take an optional 'model' kwarg (AbstractModelMaker subclass; see the 'models' submodule) that allows the user to customize how the mode profile is modelled (e.g., using a Gaussian rather than a Lorentzian)
- fit_mode now also imposes the constraint that each component of the fit should be positive.
- fit_mode now requires the central frequency of the mode to be at least a distance of gamma_max from user-specified frequency bounds

### Added
- pcko.models submodule

## 1.0

### Changed
- fit_mode and fit_mode_auto now directly take arrays of data, omega, and sigma
- fit_mode_auto: threshold is now renamed to threshold_ratio
- fit_mode_auto: Maximum number of Lorentzians is now n_lorentz_max, not n_lorentz_max - 1
- Moved from pcko.fit to pcko.utils:
	- stdev_central
	- smooth_gauss (n is now half the FWHM of the Gaussian)

### Removed
- pcko.fit.estimate_sigma
- pcko.fit._get_sigma_at_kz
- pcko.fit.smooth

Some legacy functions that were in the main namespace:

- pcko.disp_rel_from_yaver
- pcko.disp_rel_nonorm_from_yaver
- pcko.disp_rel_from_yaver_L0_HP
- pcko.disp_rel_from_dvar
- pcko.oplot_dr_f
- pcko.smooth

### Added
- pcko.getters: basic getter implementations which can be used, e.g., with get_mode_eigenfunction.

## 0.1
Initial version
