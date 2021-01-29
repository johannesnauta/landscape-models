""" Plotting """
# Necessary libraries 
import numpy as np 
import numpy.matlib as npmat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
# Import modules
import src.args 
import src.fBm

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
        suffix = "landscape_%s_%ix%i_H%.3f"%(name, 2**args.maxlevel, 2**args.maxlevel, args.H)
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
        ax.plot([L,L],[0,3*L], '--k')
        ax.plot([2*L,2*L],[0,3*L], '--k')
        ax.plot([0,3*L],[L,L], '--k')
        ax.plot([0,3*L],[2*L,2*L], '--k')
        # Limits, labels
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_xticks([0,L,2*L,3*L]);    ax.set_xticklabels([r'0', r'L', r'2L', r'3L'])
        ax.set_yticks([0,L,2*L,3*L]);    ax.set_yticklabels([r'' , r'L', r'2L', r'3L'])


    
    def plot_variance(self, args):
        # Load data
        suffix = "landscape_%ix%i_H%.3f"%(2**args.maxlevel, 2**args.maxlevel, args.H)
        Z = np.load(args.ddir+"%s.npy"%(suffix))
        K = int(1e6)
        nbins = 30

        # Define the fit 
        def fBm_Variance(d, A):
            return A*d**(2*args.H)

        # Compute variogram
        Var = src.fBm.nb_getvariogram(Z, K, nbins)
        xax = np.linspace(0, Z.shape[0]/np.sqrt(2), num=nbins)
        # Compute fit 
        popt, pcov = curve_fit(fBm_Variance, xax, Var)
        A = popt
        print(A, pcov)
        
        # Plot
        fig, ax = plt.subplots(1,1, figsize=(5,3.5), tight_layout=True)
        ax.plot(
            xax, Var, color='k', marker='o', mfc='white', 
            linestyle='none', markersize=3
        )
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
    Pjotr.plot_landscapes(args)
    # Pjotr.plot_periodiclandscape(args)
    # Pjotr.plot_variance(args)

    # Save or show 
    if args.save: 
        for name, fig in Pjotr.figdict.items():
            print("Saving... %s"%(name))
            fig.savefig("figures/%s.png"%(name), bbox_inches='tight')
    else:
        plt.show() 