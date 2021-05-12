# Mixed-Basis Reduced-Dimensional Molecular Dynamics

This GitHub repository implements the methods outlined in the paper by Morrow, Kwon, Jakubikova, and Kelley (2021). This code has been tested in Anaconda with Python 3.8.

### Requirements
- [Tasmanian](https://tasmanian.ornl.gov/), verified with v7.1
    - Must be added to `PYTHONPATH` post-install, via `export PYTHONPATH=<path-to-Tasmanian>/share/Tasmanian/python:$PYTHONPATH`
- NumPy
- SciPy

### Examples
The `examples` folder provides the code for NVE and the Langevin thermostat for mixed and all-polynomial basis choices. It is intended to facilitate reproduction of results from the paper as well as to provide a template for users wishing to use these methods on their own.

### Disclaimer
This work represents the opinions and findings of the author(s) and do not represent the official views of the National Science Foundation or the U.S. Government. This code comes with no warranty of any kind, whether expressed or implied.

