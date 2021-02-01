""" Midpoint displacement method for two-dimensional fractional Brownian motion 
    with Hurst exponent H 
"""
# Import necessary libraries
import sys, time, argparse
import numpy as np 
from numba import jit 
# Import modules 
import src.args 
import src.fBm 

    
if __name__ == "__main__":
    # Import arguments 
    Argus = src.args.Args() 
    args = Argus.args
    # Initialize class objects
    fBmAlgs = src.fBm.Algorithms(args.seed)
    # Initialize time
    starttime = time.time() 

    # Compute 2D fractional Brownian motion 
    # Z = fBmAlgs.midpointPBC2D(args.maxlevel, args.sigma, args.H)
    Z = fBmAlgs.spectral_synthesis2D(args.maxlevel, args.H, args.sigma)
    # Z = fBmAlgs.circulant_embedding2D(args.maxlevel, args.H)
    computation_time = time.time() - starttime 
    print(
        "Computation finished for %(N)ix%(N)i lattice with H=%(Hurst).3f, \
        elapsed time: %(time).4fs"%{
            'N': 2**args.maxlevel, 'Hurst': args.H, 'time': computation_time
        }
    )
    # Save 
    if args.save:
        # name = "circulant_embedding2D"
        name = "spectral_synthesis2D"
        suffix = "landscape_%s_%ix%i_H%.3f_seed%i"%(
            name, 2**args.maxlevel, 2**args.maxlevel, args.H, args.seed
        )
        np.save(args.ddir+suffix, Z)

