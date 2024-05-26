# Introduction

Utilities to plot dispersion relations and fit modes in simulations that use the Pencil code (<https://github.com/pencil-code/pencil-code>).

`power_cached` contains a replacement for `pc.read.power` that caches the results in a HDF5 file (allows faster reads and loading only a part of the array into memory)

# Dependencies
1. numpy
1. scipy
1. matplotlib
1. pencil

# Known issues
1. Using `dill` to save dr_base instances is sometimes prevented by the following bug: <https://github.com/uqfoundation/dill/issues/332> (fixed in dill 0.3.8)

# Version History

v0.1: initial version
