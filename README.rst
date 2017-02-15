===============================
OpenDeplete Depletion Front-End
===============================

This is a simple depletion front-end tool.  It currently only supports OpenMC as
the neutronics operator.  Long term goals include the addition of support for
other neutronics operators.

------------
Installation
------------

This tool is currently written against the OpenMC/develop branch Python API.  It
is also written using features of Python introduced in version 3.3.  As such,
one should follow the installation guide for
[OpenMC](https://github.com/mit-crpg/openmc) for Python 3.

The examples are currently written under the assumption that the user has
installed the NNDC data.  There is a toggle in `example_geometry.py` in
`generate_initial_number_density` to switch to MCNP6 data.

To install the OpenDeplete package, simply run:

.. code-block:: sh

   python setup.py install

Alternatively, add the root directory of the OpenDeplete package to the
`PYTHONPATH` environment variable.

-----
Usage
-----

An example script is given in three parts: `example_geometry.py`,
`example_run.py`, and `example_plot.py`.  Running `example_run.py` will load the
geometry from `example_geometry.py` and run a short depletion simulation, saving
the result into a `test` folder.  Running `example_plot.py` will then plot the
volume averaged Gd-157 concentration in the fuel pin compared to a CASMO5 run.

The depletion chain XML file can be set using either the
`opendeplete.Settings.chain_file` attribute or through the `OPENDEPLETE_CHAIN`
environment variable.

-------
License
-------

OpenDeplete is distributed under the MIT license.
