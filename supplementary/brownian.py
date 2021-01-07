import numpy as np 
import matplotlib.pyplot as plt 

# Seed
np.random.seed(42)
# Initialize
sig = 1 
nsteps = 10000

# Midpoint method for generating Brownian motion between 0 and 1
def midpointRecursion(X, lidx, hidx, level, maxlevel):
    idx = int((lidx + hidx) / 2)    
    delta = sig * 0.5**((level+1)/2)
    X[idx] = 0.5*(X[lidx]+X[hidx]) + delta * np.random.normal(0,1)
    if level < maxlevel:
        midpointRecursion(X, lidx, idx, level+1, maxlevel)
        midpointRecursion(X, idx, hidx, level+1, maxlevel)
    return X 

def midpointFractionalRecursion(X, H, lidx, hidx, level, maxlevel):
    idx = int((lidx + hidx) / 2)    
    delta = sig * 0.5**(level*H) * np.sqrt(1-2**(2*H-2))
    X[idx] = 0.5*(X[lidx]+X[hidx]) + delta * np.random.normal(0,1)
    if level < maxlevel:
        midpointFractionalRecursion(X, H, lidx, idx, level+1, maxlevel)
        midpointFractionalRecursion(X, H, idx, hidx, level+1, maxlevel)
    return X 

maxlevel = 16
N = 2**maxlevel
# H = 1/2
# X = np.zeros(N)
# X[-1] = sig * np.random.normal(0,1)
# X = midpointRecursion(X, 0, N-1, 1, maxlevel)
# Arbitrary H 
H = [0.3, 0.5, 0.7, 0.9]
end = sig * np.random.normal(0,1)
for i, h in enumerate(H):
    np.random.seed(42)
    Y = np.zeros(N)
    Y[-1] = end + i
    Y = midpointFractionalRecursion(Y, h, 0, N-1, 1, maxlevel)

    plt.plot(Y)

plt.show()


