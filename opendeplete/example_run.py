"""An example file showing how to run a simulation."""

import numpy as np

import example_geometry
import integrator
import function
import openmc_wrapper


# Load geometry from example
geometry, volume, materials, lower_left, upper_right = example_geometry.generate_problem()

# Create dt vector for 5.5 months with 15 day timesteps
dt1 = 15*24*60*60  # 15 days
dt2 = 5.5*30*24*60*60  # 5.5 months
N = np.floor(dt2/dt1)

dt = np.repeat([dt1], N)

# Create settings variable
settings = openmc_wrapper.Settings()

settings.chain_file = "../chains/chain_simple.xml"
settings.openmc_call = "openmc"
# An example for mpiexec:
# settings.openmc_call = ["mpiexec", "/home/cjosey/code/openmc/bin/openmc"]
settings.particles = 1000
settings.batches = 100
settings.inactive = 40
settings.lower_left = lower_left
settings.upper_right = upper_right
settings.entropy_dimension = [10, 10, 1]

settings.power = 2.337e15*4  # MeV/second cm from CASMO
settings.dt_vec = dt
settings.output_dir = 'test'

op = function.Operator(geometry, volume, materials, settings)

# Perform simulation using the MCNPX/MCNP6 algorithm
integrator.ce_cm(op)
