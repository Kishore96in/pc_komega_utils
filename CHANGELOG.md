# Changelog

## 1.0 (planned)

### Changed

- fit_mode and fit_mode_auto now directly take arrays of data, omega, and sigma
- fit_mode_auto: threshold is now renamed to threshold_ratio
- fit_mode_auto: Maximum number of Lorentzians is now n_lorentz_max, not n_lorentz_max - 1

#### Moved from pcko.fit to pcko.utils
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
