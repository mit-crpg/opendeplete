import function
import integrator
import numpy as np
import pickle
import utilities
import sys
import openmc_wrapper
import example_geometry

geometry, volume = example_geometry.generate_geometry()
materials = example_geometry.generate_initial_number_density()

# Create dt vector for 5.5 months with 15 day timesteps
dt1 = 15*24*60*60 # 15 days
dt2 = 5.5*30*24*60*60 # 5.5 months
N = np.floor(dt2/dt1)

dt = np.repeat([dt1], N)

# Create settings variable
settings = openmc_wrapper.Settings()

settings.cross_sections = "/home/cjosey/code/openmc/data/nndc/cross_sections.xml"
settings.chain_file = "/home/cjosey/code/opendeplete/chains/chain_simple.xml"
settings.openmc_call = "/home/cjosey/code/openmc/bin/openmc"
# An example for mpiexec:
# settings.openmc_call = ["mpiexec", "/home/cjosey/code/openmc/bin/openmc"]
settings.particles = 1000
settings.batches = 100
settings.inactive = 40

settings.power = 2.337e15 # MeV/second cm from CASMO
settings.dt_vec = dt
settings.output_dir = 'test'

op = function.Operator()
op.initialize(geometry, volume, materials, settings)

# Save results for future processing
output = open('op_MCNP.pkl', 'wb')
pickle.dump(op, output)
output.close()

import integrator
integrator.QD(op)
