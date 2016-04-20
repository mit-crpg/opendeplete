===============================
OpenDeplete Depletion Front-End
===============================

This is a simple depletion front-end tool.  It currently only supports one
geometry and only OpenMC as the neutronics operator.  Long term goals include
arbitrary geometry and OpenMOC support.

------------
Installation
------------

This tool is currently written against the OpenMC/Develop branch Python API.  It
is also written using features of Python introduced in version 3.2.  As such,
one should follow the installation guide for
[OpenMC](https://github.com/mit-crpg/openmc) for Python 3.

The code is currently coded under the assumption that the user has installed the
NNDC data.  There is a toggle in `geometry.py` in
`generate_initial_number_density` to switch to MCNP6 data.  For performance
reasons, it is highly recommended to use binary ACE files.

-----
Usage
-----

Two example scripts are given, one to run a simple problem, the other to plot
the gadolinium-157 inside of the pin.  Modify the paths inside according to the
installation directory of choice.

-------
License
-------

OpenDeplete is distributed under the MIT license.
