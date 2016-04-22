===============================
OpenDeplete Depletion Front-End
===============================

This is a simple depletion front-end tool.  It currently has rudimentary support
for multiple geometries (all cells must be in root universe), and it only
supports OpenMC as the neutronics operator.  Long term goals include
generalization of the geometry and OpenMOC support.

------------
Installation
------------

This tool is currently written against the OpenMC/Develop branch Python API.  It
is also written using features of Python introduced in version 3.2.  As such,
one should follow the installation guide for
[OpenMC](https://github.com/mit-crpg/openmc) for Python 3.

The code is currently coded under the assumption that the user has installed the
NNDC data.  There is a toggle in `example_geometry.py` in
`generate_initial_number_density` to switch to MCNP6 data.  For performance
reasons, it is highly recommended to use binary ACE files.

-----
Usage
-----

An example script is given in three parts: `example_geometry.py`,
`example_run.py`, and `example_plot.py`.  Running `example_run.py` will load the
geometry from `example_geometry.py` and run a short depletion simulation, saving
the result into a `test` folder.  Running `example_plot.py` will then plot the
volume averaged Gd-157 concentration in the fuel pin compared to a CASMO5 run.

-------
License
-------

OpenDeplete is distributed under the MIT license.
