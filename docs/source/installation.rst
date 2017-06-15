============
Installation
============

Installation is currently very complex due to the nature of the program.
OpenDeplete requires the following:
 - MPI (currently OpenMPI_ 1.10.6 and 2.1.1 are tested)
 - Python 3.3 or newer
 - mpi4py_
 - HDF5_ configured with ``--enable-parallel --enable-fortran --enable-fortran2003`` flags.  1.8.18 is tested.
 - h5py_ compiled with the above HDF5 library.
 - OpenMC_ compiled with the above HDF5 library with OpenMP and MPI enabled.
 - Nuclear data for OpenMC, preferrably NNDC to run the test suite.

.. _OpenMPI: https://www.open-mpi.org/software/ompi/
.. _mpi4py: https://pythonhosted.org/mpi4py/
.. _HDF5: https://support.hdfgroup.org/HDF5/
.. _h5py: https://github.com/h5py/h5py
.. _OpenMC: https://github.com/mit-crpg/openmc

Additionally, the OpenMC python API must be installed inside of Python 3.

Then, ensure that ``openmc`` is in the path and ``$OPENMC_CROSS_SECTIONS`` is
set.  Once all of these components are available, the program can be tested
by running the test script from the OpenDeplete root directory:

.. code-block:: sh

   mpiexec --bind-to none --map-by ppr:<number of CPU cores>:node --oversubscribe ./test.py  --suite full

``--bind-to none`` allows OpenMP to work efficiently and ``-oversubscribe``
allows both the python script and OpenMC to run simultaneously. It can be
installed by:

.. code-block:: sh

   python setup.py install --user
