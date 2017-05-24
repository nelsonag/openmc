NDPP_VERSION_MAJOR = 1
NDPP_VERSION_MINOR = 0
NDPP_VERSION = (NDPP_VERSION_MAJOR, NDPP_VERSION_MINOR)

# Check for cython availability while performing the import
try:
    import cython
except ImportError:
    print("Data pre-processing not available; try installing Cython.")

from .ndpp import *