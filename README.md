# Introduction

Utilities to plot dispersion relations and fit modes in simulations that use the Pencil code (<https://github.com/pencil-code/pencil-code>).

- `power.cached` contains a replacement for `pc.read.power` that caches the results in a HDF5 file (allows faster reads and loading only a part of the array into memory)
- `power.decimate` allows to take the output of Pencil's `power_xy` subroutine and save only a subset of it into a HDF5 file. It also contains a mixin that can be used with the classes in `read` to read the condensed HDF5 output.

# Dependencies
1. numpy
1. scipy
1. matplotlib
1. pencil
1. h5py

# Known issues
1. Using `dill` to save dr_base instances is sometimes prevented by the following bug: <https://github.com/uqfoundation/dill/issues/332> (fixed in dill 0.3.8)
