""" Plotting """
# Necessary libraries 
import numpy as np 
import numpy.matlib as npmat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
# Import modules
import src.args 
import src.fBm
import src.stats 

# Set plotting font for TeX labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# Plotting class 
class Plotter():
    def __init__(self):
        self.figdict = {}   # Dictionary used for saving figures

    def plot_landscapes(self, args):
        """ Plot landscapes containing fractional Brownian motion """ 
        # Initialize figures
        fractions = [0.15, 0.5, 0.85]
        fig, ax = plt.subplots(1, len(fractions), figsize=(3*len(fractions),3), tight_layout=True)
        # Load data
        name = "spectral_synthesis2D"
        suffix = "landscape_%s_%ix%i_H%.3f"%(name, 2**args.maxlevel, 2**args.maxlevel, args.H)
        Z = np.load(args.ddir+"%s.npy"%(suffix))
        
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
    
    def plot_periodiclandscape(self, args):
        """ Visualize the periodic boundary conditions of a landscape
            containing fractional Brownian motion 
        """
        # Initialize figures
        fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout=True)
        # Load data 
        name = "spectral_synthesis2D"
        # name = "circulant_embedding2D"
        suffix = "landscape_%s_%ix%i_H%.3f_seed%i"%(
            name, 2**args.maxlevel, 2**args.maxlevel, args.H, args.seed
        )
        Z = np.load(args.ddir+"%s.npy"%(suffix))
        
        # Determine point at which a specified fraction of the environment is filled
        f = 0.5 
        Z = Z + abs(np.min(Z))                  # Shift
        Zsort = np.sort(Z.flatten())            # Flatten & sort
        N = len(Zsort)  
        cutoff = Zsort[int((1-f)*N)]        # Find cutoff point
        Zplot = Z.copy() / cutoff           # Normalize
        Zplot[Zplot >= 1] = 1               # 
        Zplot[Zplot < 1] = 0
        # Extend to display periodic boundary conditions
        Zpbc = npmat.repmat(Zplot, 3, 3) * 0.5 
        M = 2**args.maxlevel
        Zpbc[M:2*M,M:2*M] = Zplot 
        # Plot
        ax.imshow(Zpbc, cmap='Greys', interpolation='none')
        # Visual aids
        L = M - 1/2
        ax.plot([L,L],[0,3*L], '--k', linewidth=0.75)
        ax.plot([2*L,2*L],[0,3*L], '--k', linewidth=0.75)
        ax.plot([0,3*L],[L,L], '--k', linewidth=0.75)
        ax.plot([0,3*L],[2*L,2*L], '--k', linewidth=0.75)
        # Limits, labels
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_xticks([0,L,2*L,3*L]);    ax.set_xticklabels([r'0', r'L', r'2L', r'3L'])
        ax.set_yticks([0,L,2*L,3*L]);    ax.set_yticklabels([r'' , r'L', r'2L', r'3L'])
        ax.text(
            0.5, 1.05, r"$f=%.1f, H=%.1f$"%(f, args.H), ha='center', fontsize=15,
            transform=ax.transAxes
        )
        # Save
        suffix = "_f%.1f_H%.1f"%(f, args.H)
        self.figdict[name+suffix] = fig 

    def plot_variance1D(self, args):
        """ Plot the variance of a 1D Brownian motion """ 
        # Instantiate objects
        Alg = src.fBm.Algorithms(args.seed) 
        # Allocate
        Var = np.zeros(args.nbins)
        # Initialize 
        N = 2**args.maxlevel
        for n in range(args.n): 
            # Generate Brownian motion 
            X = Alg.Brownian(N)
            Var_, Dist = src.stats.nb_getvariance(X, args.nbins, args.K)
            Var += Var_ 
        Var /= args.n 
        # Initialize figure 
        fig, ax = plt.subplots(1,1, figsize=(5,2.5), tight_layout=True)
        ax.plot(
            Dist, Var, color='k', marker='o', linestyle='none', mfc='white',
            markersize=3
        )
        # Define the fit 
        def fBm_Variance(d, A):
            return A*d
        # Compute best fit
        popt, pcov = curve_fit(fBm_Variance, Dist, Var)
        A = popt
        print(A)
        ax.plot(
            Dist, fBm_Variance(Dist, A), color='k',
            label=r"$f(t-s) \propto (t-s)^{2H}$"
        )


    
    def plot_variogram2D(self, args):
        # Load data
        N = 2**args.maxlevel
        suffix = "_%ix%i_H%.3f"%(N, N, args.H)
        Var = np.load(args.ddir+"variogram_spectral_synthesis2D%s.npy"%(suffix))
        # Var = np.load(args.ddir+"variogram_circular_embedding2D%s.npy"%(suffix))
        meanVar = np.mean(Var, axis=1) #**(2*args.H)
        delta = np.load(args.ddir+"distances%s.npy"%(suffix))

        # Define the fit 
        def fBm_Variance(d, A):
            return A*d**(2*args.H)
        popt, pcov = curve_fit(fBm_Variance, delta, meanVar)
        A = popt
        print(A, pcov)
        
        # Plot
        fig, ax = plt.subplots(1,1, figsize=(5,3.5), tight_layout=True)
        ax.plot(
            delta, meanVar, color='k', marker='o', mfc='white', 
            linestyle='none', markersize=3
        )
        xax = np.linspace(0, np.max(delta), 10*len(delta))
        ax.plot(
            xax, fBm_Variance(xax, A), color='k',
            label=r"$f(t-s) \propto (t-s)^{2H}$"
        )
        # Limits, labels, etc 
        ax.set_ylabel(r"$\hat{\gamma}\big(|X(x)-X(x+d)|^2\big)$")
        ax.legend(fontsize=11, loc='lower right')



if __name__ == "__main__":
    # Import arguments 
    Argus = src.args.Args() 
    args = Argus.args 

    # Plot 
    Pjotr = Plotter()
    # Pjotr.plot_landscapes(args)
    Pjotr.plot_periodiclandscape(args)
    # Pjotr.plot_variance1D(args)
    # Pjotr.plot_variogram2D(args)

    # Save or show 
    if args.save: 
        for name, fig in Pjotr.figdict.items():
            print("Saving... %s"%(name))
            fig.savefig("figures/%s.png"%(name), bbox_inches='tight')
    else:
        plt.show() 