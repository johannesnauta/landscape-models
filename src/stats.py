""" Holds methods for statistical computations """ 
# Import necessary libraries
import sys 
import numpy as np 
from numba import jit 
# Import methods 
sys.path.append('../')
import src.fBm 
import src.args

#################
# Numba methods # 
#################
@jit(nopython=True, cache=True)
def nb_getvariance(X, nbins, K):
    """ Approximate the variance of the increments 
        Does not assume periodic boundary conditions 
    """ 
    N = len(X)
    maxdist = N // 3
    distances = np.array([i*maxdist for i in range(nbins)])
    # Initialize
    MeanVar = np.zeros(nbins)
    for b in range(1,nbins):
        sq_increments = np.zeros(K)
        dx = b*np.int64(maxdist / nbins)
        valid_idx = np.array([i for i in range(dx,N-dx)])
        K = min(K, len(valid_idx))
        idx = np.random.choice(valid_idx, size=K)
        for k in range(K):
            # Sample random lattice point 
            i = idx[k]
            sq_increments[k] = abs(X[i+dx]-X[i])**2
        MeanVar[b] = np.mean(sq_increments)
    return MeanVar, distances

@jit(nopython=True, cache=True)
def nb_estimate_variogram(X, nbins, M, K):
    """ Estimate the variogram by random sampling a subset of M sample points,
        for which the increments with K other random points will be measured.
        To compute the variogram, the squared absolute difference is taken, 
        of which the mean reflects the variance of the increments.

        Assumes 2D random field
        Assumes periodic boundary conditions (PBC)
    """
    np.random.seed(nbins)
    N, _ = X.shape 
    # Create array of ring centers 
    maxdist = N / 5
    ringwidth = maxdist / nbins 
    # Construct the matrix of distances between lattice points and the upper left
    # lattice point. When sampling a random point, one can always shift the lattice
    # such that the sampled point is the upper left element. This allows us to 
    # compute this distance matrix only once, and shift relevant indices by the same
    # shift needed to make the appropriate index (i,j) to be (0,0). The matrix is
    # furthermore symmetrical, so we can compute only the upper (or lower) elements.
    #
    # Construct lattice
    lattice = np.array([[i,j] for i in range(N) for j in range(N)])
    # Compute distance with (0,0) at lattice[0,0]
    absdist = np.abs(lattice[0,0] - lattice)
    absdist = np.where(absdist > 0.5*N, absdist-N, absdist)             # PBC
    distances = np.sqrt(np.sum(absdist**2, axis=1))
    
    # Allocate 
    variogram = np.zeros(nbins, dtype=np.float32)
    ring_centres = np.zeros(nbins, dtype=np.float32)
    for b in range(nbins):
        # Allocate
        sq_difference = np.zeros(M*K)   
        # Compute ring edges 
        hmin = b*ringwidth
        hmax = (b+1)*ringwidth
        ring_centres[b] = hmin + (hmax - hmin) / 2 
        # Gather indices in the ring, 
        # i.e. indices for which hmin < distance <= hmax 
        indices = np.where((distances>hmin)*(distances<=hmax))[0]
        test = np.zeros((N,N))
        test_ = np.zeros(N*N)
        # When distances are small, it can happen that not many points are
        # around, so clip the number of sampled increments per point K 
        K = min(K, len(indices))
        # Sample M different random points from X 
        rand_idxs = np.random.choice(np.arange(N*N), size=M, replace=False)
        c = 0   # Index counter for square differences
        for m in range(M): 
            shifted_indices = np.zeros(len(indices), dtype=np.int64)
            # Sample random lattice index
            i0, j0 = lattice[rand_idxs[m]]
            # Shift other indices in ring accordingly
            for t, idx in enumerate(indices):
                i1, j1 = lattice[idx] 
                # Shift 
                i1 = (i1 + i0) % N
                j1 = (j1 + j0) % N 
                shifted_indices[t] = i1*N + j1
            # Sample K random points from the ring (if possible)
            if K > 0: 
                samp_idx = shifted_indices[np.random.choice(np.arange(K), size=K, replace=False)]
                # samp_idx = indices.copy()
                # Compute squared absolute difference 
                for idx_ in samp_idx:
                    i1, j1 = lattice[idx_]
                    dist_x = abs(i0-i1) 
                    dist_x = N - dist_x if dist_x > 0.5*N else dist_x
                    dist_y = abs(j0-j1) 
                    dist_y = N - dist_y if dist_y > 0.5*N else dist_y 
                    dist = np.sqrt(dist_x**2 + dist_y**2)
                    # print(dist, hmin, hmax)
                    sq_difference[c] = abs(X[i0,j0] - X[i1,j1])**2
                    c += 1 
            else: 
                # No data points at specified distance
                # If this happens, choose bins better 
                print("No data points at specified distance, please check parameters")
        
        # Compute the variogram as the mean of squared absolute differences 
        variogram[b] = np.mean(sq_difference)
    return variogram, ring_centres
        

if __name__ == "__main__":
    # Load arguments
    Args = src.args.Args() 
    args = Args.args
    # Load class for generating fBm 
    fBmGen = src.fBm.Algorithms(args.seed)
    variogram = np.zeros((args.nbins, args.n))
    for i in range(args.n):
        # X = fBmGen.spectral_synthesis2D(args.maxlevel, args.H)
        X = fBmGen.circulant_embedding2D(args.maxlevel, args.H)
        Var, Dist = nb_estimate_variogram(X, args.nbins, args.K, 10*args.K)
        variogram[:,i] = Var
    # Save 
    N = 2**args.maxlevel
    suffix = "_%ix%i_H%.3f"%(N,N, args.H)
    # np.save("../data/variogram_spectral_synthesis2D%s"%(suffix), variogram)
    np.save("../data/variogram_circular_embedding2D%s"%(suffix), variogram)
    np.save("../data/distances%s"%(suffix), Dist)

