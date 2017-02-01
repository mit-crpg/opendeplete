"""An example file showing how to plot data from a simulation."""

import matplotlib.pyplot as plt
import numpy as np

from opendeplete import load_directory, evaluate_result_list

# Set variables for where the data is, and what we want to read out.
result_folder = "test"

# Load data
results = load_directory(result_folder)

x, y = evaluate_result_list(results, 0, use_interpolation=False)

# Plot data
plt.semilogy(x, y["10004"]["Gd157"], label="Pointwise")
x, y = evaluate_result_list(results, 1000, use_interpolation=True)

# Plot data
plt.semilogy(x, y["10004"]["Gd157"], label="C1 Continuous")
plt.legend(loc="best")
plt.savefig("interp.pdf")
