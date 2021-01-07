""" Midpoint displacement method for two-dimensional fractional Brownian motion 
    with Hurst exponent H 
"""
import sys, time, argparse
import numpy as np 
from numba import jit 

import matplotlib.pyplot as plt 
from matplotlib import cm

##############
# Numba code #
##############
@jit(nopython=True, cache=True)
def nb_seed(seed):
    """ Ensure that numba is seeded as well """
    np.random.seed(seed)

@jit(nopython=True, cache=True)
def nb_midpoint(X, maxlevel, sigma, H):
    """ Numba implementation of midpoint displacement 
        Implements the midpoint displacement method for fBm with an 
        arbitrary Hurst parameter H
        For details on the algorithm, see: Saupe, 1988, Algorithms for random fractals
    """ 
    N = 2**maxlevel
    # Initialize 
    delta = sigma       # Variance 
    # Initialize corners
    X[0,0] = delta * np.random.normal(0,1) 
    X[0,-1] = delta * np.random.normal(0,1)
    X[-1,0] = delta * np.random.normal(0,1)
    X[-1,-1] = delta * np.random.normal(0,1)
    # Initialize strided index variables
    D = N 
    d = N // 2

    # Precompute the Gaussian distributed noise values 
    noise = np.random.normal(0, 1, size=(N+1,N+1))
    
    # Run the midpoint method for increasing resulution
    for level in range(maxlevel): 
        # Update delta
        delta = delta * (1/2)**(0.5*H)
        # Diagonal grid 
        for i in range(d, N-d+1, D):
            for j in range(d, N-d+1, D):                    
                X[i,j] = ( X[i+d,j+d] + X[i-d,j+d] + X[i+d,j-d] + X[i-d,j-d] ) / 4 + delta * noise[i,j]
        # Update delta
        delta = delta * (1/2)**(0.5*H)
        # Interpolate and offset boundary grid points
        for i in range(d, N-d+1, D):
            X[i,0] = ( X[i+d,0] + X[i-d,0] + X[i,d] ) / 3 + delta * noise[i,0] 
            X[i,-1] = ( X[i+d,-1] + X[i-d,-1] + X[i,N-d-1] ) / 3 + delta * noise[i,-1] 
            X[0,i] = ( X[0,i] + X[0,i-d] + X[d,i] ) / 3 + delta * noise[0,i] 
            X[-1,i] = ( X[-1,i+d] + X[-1,i-d] + X[N-d-1,i] ) / 3 + delta * noise[-1,i] 
        # Update square grid
        for i in range(d, N-d+1, D):
            for j in range(D, N-d+1, D):
                X[i,j] = ( X[i,j+d] + X[i,j-d] + X[i+d,j] + X[i-d,j] ) / 4 + delta * noise[i,j]
        for i in range(D, N-d+1, D):
            for j in range(d, N-d+1, D):
                X[i,j] = ( X[i,j+d] + X[i,j-d] + X[i+d,j] + X[i-d,j] ) / 4 + delta * noise[i,j]
        # Update strided index variables
        D = D // 2 
        d = d // 2
    return X 

class fractionalBrownianMotion():
    def __init__(self, seed):
        self.d = 2 
        self.seed = seed
    
    def midpoint(self, maxlevel, sigma, H):
        """ Midpoint method for computing fBm """
        ## Precomputations before calling numba implementation
        # Initialize RNG
        np.random.seed(self.seed)
        nb_seed(self.seed)
        # Initialize lattice
        N = 2**args.maxlevel
        X = np.zeros((N+1,N+1))
        # Compute fBm using the midpoint method
        Z = nb_midpoint(X, maxlevel, sigma, H)
        return Z 

    
if __name__ == "__main__":
    # Extract arguments 
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '--m', dest='maxlevel', type=int, default=8,
        help='maximum level of resolution as N=2**maxlevel'
    )
    parser.add_argument(
        '--sig', dest='sigma', type=float, default=1.,
        help='variance of the Gaussian distribution'
    )
    parser.add_argument(
        '--H', dest='H', type=float, default=0.5, help='Hurst exponent'
    )
    parser.add_argument(
        '--seed', dest='seed', type=int, default=42
    )
    parser.add_argument(
        '--save', dest='save', action='store_true',
        help='if included, saves the figure'
    )
    args = parser.parse_args()
    # Initialize time
    starttime = time.time() 

    # Initialize fBm class object
    fBm = fractionalBrownianMotion(args.seed) 
    # Compute 2D fractional Brownian motion 
    Z = fBm.midpoint(args.maxlevel, args.sigma, args.H)
    computation_time = time.time() - starttime 
    print("Computation finished for %(N)ix%(N)i lattice with H=%(Hurst).3f, elapsed time: %(time).4fs"%{
        'N': 2**args.maxlevel+1, 'Hurst': args.H, 'time': computation_time}
    )

    # Plot
    fractions = [0.15, 0.5, 0.85]
    fig, ax = plt.subplots(1, len(fractions), figsize=(3*len(fractions),3), tight_layout=True)
    # Determine point at which a specified fraction of the environment is filled
    Z = Z + abs(np.min(Z))                  # Shift
    Zsort = np.sort(Z.flatten())            # Flatten & sort
    N = len(Zsort)  
    for i, f in enumerate(fractions):
        cutoff = Zsort[int((1-f)*N)]        # Find cutoff point
        Zplot = Z.copy() / cutoff           # Normalize
        Zplot[Zplot >= 1] = 1               # 
        Zplot[Zplot < 1] = 0
        # Plot
        ax[i].imshow(Zplot, cmap='Greys', interpolation='none')
        # Limits, labels, etc
        ax[i].text(0.5,1.05, r'f=%.2f'%(f), ha='center', fontsize=11, transform=ax[i].transAxes)
        if i == 0:
            ax[i].set_ylabel(r'H=%.2f'%(args.H), fontsize=11)
        L = np.sqrt(N)-1
        ax[i].set_xlim(0, L)
        ax[i].set_ylim(0, L)
        ax[i].set_xticks([0,L]);    ax[i].set_xticklabels([r'0', r'L'])
        ax[i].set_yticks([0,L]);    ax[i].set_yticklabels([r'', r'L'])

    if args.save:
        fig.savefig('figures/landscape_H%.2f.png'%(args.H))
    else:
        plt.show()

