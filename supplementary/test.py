import numpy as np 
import matplotlib.pyplot as plt 
from fbm import FBM 

level = 10
N = 2**level

np.random.seed(42)
f = FBM(N, hurst=0.5)

K = 1000
nbins = 50
h = np.zeros(nbins)
for k in range(K):
    X = f.fbm() 
    h_, _ = np.histogram(X, bins=nbins, range=(-1,1), density=True)
    h += h_ 

h /= K
t_values = f.times() 

fig, ax = plt.subplots(1,1, figsize=(5,3.5), tight_layout=True) 
ax.plot(h)
plt.show()
