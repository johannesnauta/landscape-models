""" Collect arguments """
import argparse
import numpy as np 

class Args():
    def __init__(self):
        parser = argparse.ArgumentParser("Specify specific variables")
        ## Add arguments
        # Algorithm variables
        parser.add_argument(
            '--m', dest='maxlevel', type=int, default=9,
            help='maximum level of resolution as N=2**maxlevel'
        )
        parser.add_argument(
            '--sig', dest='sigma', type=float, default=1.,
            help='variance of the Gaussian distribution'
        )
        parser.add_argument(
            '--H', dest='H', type=float, default=0.5, help='Hurst exponent'
        )
        # Random number variables
        parser.add_argument(
            '--seed', dest='seed', type=int, default=1
        )
        # Plotting variables 
        parser.add_argument(
            '--n', dest='n', type=int, default=100,
            help='number of different landscapes to generate'
        )
        parser.add_argument(
            '--K', dest='K', type=int, default=200,
            help='number of samples for computing averages'
        )
        parser.add_argument(
            '--nbins', dest='nbins', type=int, default=25,
            help='number of different distances at which to compute absolute difference'
        )
        # Boolean variables
        parser.add_argument(
            '--save', dest='save', action='store_true',
            help='if included, saves the figure'
        )
        # Directory variables 
        parser.add_argument(
            '--ddir', dest='ddir', type=str, default='data/',
            help='specify directory for output data'
        )
        
        # Parse arguments
        self.args = parser.parse_args()