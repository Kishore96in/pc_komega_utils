Utilities to plot dispersion relations and fit modes in simulations that use the Pencil code (<https://github.com/pencil-code/pencil-code>).

# Dependencies
1. numpy
1. scipy
1. matplotlib
1. pencil

# Known issues
1. Using `dill` to save dr_base instances is sometimes prevented by the following bug: <https://github.com/uqfoundation/dill/issues/332> (fixed in dill 0.3.8)
