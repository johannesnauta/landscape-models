""" Implements methods for generating fraction Brownian motion (fBm)

    Currently support only 1- and 2-dimensional systems 
    Uses Numba for efficient code 
"""
# Import necessary libraries
import numpy as np 
from numba import jit 
import numpy.matlib as npmat

#################
# Numba methods #
#################
@jit(nopython=True, cache=True)
def nb_seed(seed):
    """ Ensure that numba is seeded as well """
    np.random.seed(seed)

###################
## Generating fBm #
@jit(nopython=True, cache=True)
def nb_midpoint1D(maxlevel, sigma, H):
    N = 2**maxlevel 
    # Initialize 
    delta = sigma * np.sqrt(0.5) * np.sqrt(1-2**(2*H-2))      # Variance 
    # Initialize
    X = np.zeros(N+1)
    X[0] = 0 
    X[-1] = sigma * np.random.normal(0,1)
    # X[-1] = X[0]
    # Initialize strided index variables 
    D = N 
    d = N // 2 
    # Precompute Gaussian noise values 
    noise = np.random.normal(0, 1, N) 

    # Run midpoint method 
    for level in range(maxlevel):
        # delta = delta * 0.5 ** ((level-1) * H)
        delta = delta * 0.5**((level+1)*H)
        for i in range(d, N-d+1, D):
            X[i] = (X[i-d] + X[i+d]) / 2 #+ delta * np.random.normal(0,1)
        for i in range(0, N, d):
            X[i] = X[i] + delta * np.random.normal(0, 1)
        D = D // 2
        d = d // 2 
    
    return X 

@jit(nopython=True, cache=True)
def nb_midpoint2D(maxlevel, sigma, H):
    """ Numba implementation of midpoint displacement 
        Implements the midpoint displacement method for fBm with an 
        arbitrary Hurst parameter H
        For details on the algorithm, see: Saupe, 1988, Algorithms for random fractals
    """ 
    N = 2**maxlevel
    # Initialize 
    delta = sigma           # Variance 
    X = np.zeros((N+1,N+1)) # Lattice
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
            X[0,i] = ( X[0,i+d] + X[0,i-d] + X[d,i] ) / 3 + delta * noise[0,i] 
            X[-1,i] = ( X[-1,i+d] + X[-1,i-d] + X[N-d-1,i] ) / 3 + delta * noise[-1,i] 
        # Update square grid
        for i in range(d, N-d+1, D):
            for j in range(D, N-d+1, D):
                X[i,j] = ( X[i,j+d] + X[i,j-d] + X[i+d,j] + X[i-d,j] ) / 4 + delta * noise[i,j]
        for i in range(D, N-d+1, D):
            for j in range(d, N-d+1, D):
                X[i,j] = ( X[i,j+d] + X[i,j-d] + X[i+d,j] + X[i-d,j] ) / 4 + delta * noise[i,j]
        # Update strided index variables
        print(X); exit()
        D = D // 2 
        d = d // 2
    return X 

@jit(nopython=True, cache=True)
def nb_midpointPBC2D(maxlevel, sigma, H):
    """ Numba implementation of midpoint displacement method with periodic
        boundary conditions (PBC), with arbitrary Hurst parameter H 
    """ 
    N = 2**maxlevel         # Size (resolution) of the grid 
    # Initialize 
    delta = sigma           # Variance for each resolution level 
    # Initialize corners periodically 
    X = np.zeros((N, N), dtype=np.float32)
    X[0,0] = delta * np.random.normal(0,1) 
    # Initialize strided index variables 
    D = N 
    d = N // 2 

    # Precompute Gaussian noise values 
    noise = np.random.normal(0,1, size=(N,N))

    # Run midpoint method 
    for level in range(maxlevel):
        # Update delta 
        delta = delta * (1/2)**(H / 2)
        # Diagonal grid 
        for i in range(d, N-d+1, D):
            for j in range(d, N-d+1, D):
                X[i,j] = (
                    X[(i+d)%N, (j+d)%N] +
                    X[(i-d)%N, (j+d)%N] +
                    X[(i+d)%N, (j-d)%N] +
                    X[(i-d)%N, (j-d)%N] 
                ) / 4 + delta * noise[i,j]
        # Update delta 
        delta = delta * (1/2)**(H / 2)
        # Update boundary grid points 
        for i in range(d, N-d+1, D):
            X[i,0] = (
                X[(i+d)%N, 0] + 
                X[(i-d)%N, 0] + 
                X[i, d] + 
                X[i, (-d)%N]
            ) / 4 + delta * noise[i,0]
            X[0,i] = (
                X[0, (i+d)%N] + 
                X[0, (i-d)%N] + 
                X[d,i] + 
                X[(-d)%N, i]
            ) / 4 + delta * noise[0,i]
        # Update square grid 
        for i in range(d, N-d+1, D):
            for j in range(D, N-d+1, D):
                X[i,j] = ( 
                    X[i,(j+d)%N] + 
                    X[i,(j-d)%N] + 
                    X[(i+d)%N,j] + 
                    X[(i-d)%N,j] 
                ) / 4 + delta * noise[i,j]
        for i in range(D, N-d+1, D):
            for j in range(d, N-d+1, D):
                X[i,j] = ( 
                    X[i,(j+d)%N] + 
                    X[i,(j-d)%N] + 
                    X[(i+d)%N,j] + 
                    X[(i-d)%N,j] 
                ) / 4 + delta * noise[i,j]
        # Update strided index variables
        D = D // 2 
        d = d // 2
    return X

class Algorithms():
    """ Class object that holds algorithms for generating fractional Brownian motion
        Uses the Numba functions, but allows for pre-computations if necessary, so 
        kind of acts as a wrapper for the efficient Numba implementations
    """
    def __init__(self, seed):
        self.seed = seed 

    def Brownian(self, N, sig=1):
        x = np.zeros(N) 
        noise = np.random.normal(0,sig,N-1)
        x[1:] = np.cumsum(noise) 
        return x 

    def midpoint2D(self, maxlevel, sigma, H):
        return nb_midpoint2D(maxlevel, sigma, H)
    
    def midpointPBC2D(self, maxlevel, sigma, H):
        return nb_midpointPBC2D(maxlevel, sigma, H)

    def spectral_synthesis1D(self, N, H):
        """ Generates fractal Brownian motion in 1D, using circulant embedding approach
            see: Kroese & Botev (2015), Spatial process generation
            (https://arxiv.org/pdf/1308.0399v1.pdf)
        """
        r = np.zeros(N+1)
        r[0] = 1
        for k in range(1,N+1):
            r[k] = 0.5*((k+1)**(2*H) - 2*k**(2*H) + (k-1)**(2*H))
        rolled = np.roll(r[::-1],-1)[:-2]
        r_ = np.concatenate((r,rolled))
        # Compute eigenvalues 
        lambda_ = np.real(np.fft.fft(r_)) / (2*N)
        # Construct xomplect Gaussian vector 
        Z = np.random.normal(0, 1, 2*N) + 1j * np.random.normal(0, 1, 2*N) 
        W = np.fft.fft(np.sqrt(lambda_) * Z) 
        # Rescale 
        W = N**(-H) * np.cumsum(np.real(W))
        return W 

    def spectral_synthesis2D(self, level, H, sig=1, bounds=[0,1]):
        """ Generates fractional Brownian motion in 2D 
            For details on the algorithm, see: Saupe, 1988, Algorithms for random fractals
        """
        N = 2**level
        A = np.zeros((N,N), dtype=np.cdouble)

        for i in range(N//2):
            for j in range(N//2):
                phase = 2*np.pi*np.random.random() 
                if i != 0 or j != 0:
                    r = (i*i+j*j)**(-(H+1)/2) * np.random.normal(0,sig)
                else:
                    r = 0 
                A[i,j] = r*np.cos(phase) + 1j*r*np.sin(phase)
                
                i0 = 0 if i == 0 else N-i 
                j0 = 0 if j == 0 else N-j 
                A[i0,j0] = r*np.cos(phase) + 1j*r*np.sin(phase) 

        for i in range(1,N//2):
            for j in range(1,N//2):
                phase = 2*np.pi*np.random.random() 
                r = (i*i+j*j)**(-(H+1)/2) * np.random.normal(0,sig)
                A[i,N-j] = r*np.cos(phase) + 1j*r*np.sin(phase)
                A[N-i,j] = r*np.cos(phase) - 1j*r*np.sin(phase) 

        X = np.real(np.fft.fft2(A))
        return X 
    
    def verify_statistics(self, K, level, H, sig=1, bounds=[0,1], nbins=10):
        """ Verify that the implementations of fBm are statistically sound """ 
        # Initialize
        L = 2**level 
        # Allocate
        Var = np.zeros((K,nbins))
        for k in range(K):
            # Generate fBm 
            X = self.spectral_synthesis2D(level, H, sig=sig, bounds=bounds)
            variogram, distance_ax = nb_getvariogram(X, min(1000,2**level), nbins)
            Var[k,:] = variogram
        
        return np.mean(Var,axis=0), distance_ax
            




if __name__ == "__main__":
    maxlevel = 7
    sigma = 1 
    H = 0.9
    import matplotlib.pyplot as plt 
    # X = nb_midpoint1D(maxlevel, sigma, H)
    # X = nb_spectral_synthesis2D(2**maxlevel, H)
    fig, ax = plt.subplots(1,1, figsize=(5,3.5), tight_layout=True)
    Xplot = npmat.repmat(X, 3, 3) 
    L = 2**maxlevel
    ax.plot([L,L],[0,3*L], '--k')
    ax.plot([2*L,2*L],[0,3*L], '--k')
    ax.plot([0,3*L],[L,L], '--k')
    ax.plot([0,3*L],[2*L,2*L], '--k')
    ax.imshow(Xplot)

    plt.show()