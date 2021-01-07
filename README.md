# Landscape models 
Landscape models for use in styding (collective) foraging and/or other research.

## Midpoint method for 2D fractional Brownian motion 
The relevant file to run is `midpointFBM2D.py`.
It depends on the standard `numpy` and `matplotlib` libraries, but also on the `numba` library that enables blazingly fast calculations in Python. 
Simply installing these libraries through `pip` and/or `conda` should work. 
Note that `numba` compiles the Python code, and therefore the initial run should take (a bit) more time. 
Subsequent executions should take much less time to execute. 

Default parameters are sensible, but can be tuned by including flags when you run the code. 
Please run `python midpointFBM2D.py --help` for more details.

### Some results
![H0-15](figures/landscape_H0.15.png)
![H0-50](figures/landscape_H0.50.png) 
![H0-85](figures/landscape_H0.85.png) 