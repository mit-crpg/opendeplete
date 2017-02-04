"""An example file showing how to run a simulation."""

import numpy as np
import opendeplete

import dummy_geometry

# Create settings variable
settings = opendeplete.Settings()

settings.chain_file = "../chains/chain_simple.xml"

settings.dt_vec = [1.5]
settings.output_dir = 'test'
settings.tol = 1.0e-5

op = opendeplete.Operator()
op.geometry = DummyGeometry()

# Perform simulation using the MCNPX/MCNP6 algorithm
opendeplete.integrate(op, opendeplete.ce_cm_c1)
