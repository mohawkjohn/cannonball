import unittest
from test import test_orbiter

def cannonball_test_suite():
    """
    Load unit tests from each file for automatic running using 
    `python setup.py test`.
    """
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_orbiter))

    return suite


