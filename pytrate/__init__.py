__description__ = \
"""
Python package for performing fits of arbitrarily complex models to titration experiments.
"""
__author__ = ""
__date__ = ""
__all__ = ["experiments","indiv_models","global_connectors","util"]

from . import experiments
from . import indiv_models
from . import global_connectors
from . import fitters
from . import util

# Load these defaults (which will be used for vast majority of experiments) 
# into the global namespace for convenience
from .experiments import BaseExperiment
from .global_fit import GlobalFit
from .global_connectors import GlobalConnector

