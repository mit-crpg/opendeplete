#!/usr/bin/env python3
""" Runs opendeplete's test suite.

There are two test suites:
 1. The "normal" test suite contains all tests that take less than a second.
    This excludes OpenMC tests.  This is default.
 2. The "full" test suite contains all in the "normal" suite, as well as
    A few tests of the OpenMC functionality as well.  Passing this test
    basically guarantees everything works fully coupled together, but
    it can take a few minutes.

The test suite is passed as the first argument.
"""

import unittest
import argparse

# Tests.  Add them as they're produced.

suite_normal = [
    "test.test_depletion_chain",
    "test.test_integrator",
    "test.test_integrator_coeffs",
    "test.test_integrator_regression",
    "test.test_nuclide",
    "test.test_reaction_rates"
    ]

suite_full = [
    "test.test_full"
    ]

def test(use_full):
    """ Run all tests in suite.

    Parameters
    ----------
    use_full : bool
        Whether or not to do tests listed in suite_full.
    """

    test_suite = unittest.TestSuite()

    for module_test in suite_normal:
        tests = unittest.defaultTestLoader.loadTestsFromName(module_test)
        test_suite.addTest(tests)

    if use_full:
        for module_test in suite_full:
            tests = unittest.defaultTestLoader.loadTestsFromName(module_test)
            test_suite.addTest(tests)

    unittest.TextTestRunner().run(test_suite)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Runs opendeplete's test suite.")

    parser.add_argument("--suite", type=str, default="normal",
                        help="Which suite to run, \"normal\" or \"full\", (default: \"normal\")")

    args = parser.parse_args()

    full_test = bool(args.suite == "full")

    test(full_test)
