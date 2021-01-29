import numpy as np 
import matplotlib.pyplot as plt 


class Generate():
    def __init__(self):
        pass

    def Brownian(self, X0, T, dt):
        """ Brownian motion for T steps, with discrete time steps dt 
        """ 
        X = np.zeros(T)
        X[0] = X0 
        noise = np.random.normal(0, 1, size=T-1)
        X[1:] = X0 + np.sqrt(dt) * np.cumsum(noise)
        return X 
    
    def midpointRecursion(self, X, lidx, hidx, level, maxlevel):
        """ Midpoint method for generating Brownian motion between 0 and 1
        """
        idx = int((lidx + hidx) / 2)    
        delta = sig * 0.5**((level+1)/2)
        X[idx] = 0.5*(X[lidx]+X[hidx]) + delta * np.random.normal(0,1)
        if level < maxlevel:
            self.midpointRecursion(X, lidx, idx, level+1, maxlevel)
            self.midpointRecursion(X, idx, hidx, level+1, maxlevel)
        return X 

    def midpointFractionalRecursion(self, X, H, lidx, hidx, level, maxlevel):
        """ Midpoint method for generating fractional Brownian motion between 0 and 1 
        """
        idx = int((lidx + hidx) / 2)    
        delta = sig * 0.5**(level*H) * np.sqrt(1-2**(2*H-2))
        X[idx] = 0.5*(X[lidx]+X[hidx]) + delta * np.random.normal(0,1)
        if level < maxlevel:
            self.midpointFractionalRecursion(X, H, lidx, idx, level+1, maxlevel)
            self.midpointFractionalRecursion(X, H, idx, hidx, level+1, maxlevel)
        return X 


if __name__ == "__main__":
    # Seed
    # Initialize
    sig = 1 
    nsteps = 1000
    maxlevel = 16
    N = 2**maxlevel
    dt = 1
    r = 10
    # Instantiate
    fBm = Generate()

    np.random.seed(42)
    X = fBm.Brownian(0, nsteps, dt)
    np.random.seed(42)
    Y = fBm.Brownian(0, nsteps, r*dt) / np.sqrt(r)

    plt.plot(X)
    plt.plot(Y)
    plt.show()

    exit()
    # Fractional Brownian motion 
    # Arbitrary H 
    H = [0.3, 0.5, 0.7, 0.9]
    end = sig * np.random.normal(0,1)
    for i, h in enumerate(H):
        np.random.seed(42)
        Y = np.zeros(N)
        Y[-1] = end + i
        Y = fBm.midpointFractionalRecursion(Y, h, 0, N-1, 1, maxlevel)

        plt.plot(Y)

    plt.show()


