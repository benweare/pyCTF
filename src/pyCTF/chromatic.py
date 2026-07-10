import numpy as np 
import matplotlib.pyplot as plt 

from pyCTF.utils import gradient_simple


# Contains two methods for measureing Cc (lmfit and numpy).
class chromaticAberration:
    '''
    Class for chromatic aberration measurement.

    Attributes
    ----------
    kV : float
        Accelerating voltage.
    focusSeries : array-like
        Defocus values.
    voltage Series : array-like
        Accelerating voltage values.
    results : class
        lmfit ModelResults class. 
    cov : array-like
        Covariance.
    intercept : float
        y-intercept of fit.
    slope : float
        Gradient of fit.

    Methods
    ------
    fit()
        Fits the gradient using y=mx+c. 
    print_results()
        Print results of simple fitting.
    plot_figure()
        Plot results of fitting.

    Notes
    -----
    This class contains methods to find chromatic aberration based on how
    focus changes as a function of accelerating voltage. It would also be
    possible to measure based on how it changes as a function of lens current.

    Data should be supplied in the following format:
    Array of voltages e.g. np.array([200.0, 199.95, 199.90, 199.85, 199.80, 199.75, 199.70])
    Array of defocuses e.g. np.array([ -714.09, -433.04, -197.66, 0, 240.20, 458.89, 688.94 ])*1e-9
    
    '''

    def __init__( self, kV, voltage, defocus ):
        '''
        Parameters
        ----------
        kV : float
            Divisor, accelerating voltage. 
        voltage : array-like
            Dividend, array of accelerating voltages.
        defocus : array-like
            Array of defocus values.

        Attributes
        ----------
        results : class
            lmfit ModelResults class. 
        cov : array-like
            Covariance.
        intercept : float
            y-intercept of fit.
        slope : float
            Gradient of fit.
        '''
        self.kV = kV
        self.focus_series = defocus
        self.voltage_series = voltage
        self.results = None
        # for simple gradient
        self.cov = None
        self.intercept = None
        self.slope = None
        return

    def fit( self, **kwargs ):
        '''
        Fit data. 
        '''
        method = kwargs.get( 'method', 'numpy' )
        if ( method == 'numpy' ):
            self._fit_simple()
        if ( method == 'lmfit' ):
            self._fit_lmfit()
        return

    # First order polynomial fitting  using numpy.
    def _fit_simple( self ):
        from numpy.polynomial import polynomial as P
        # gradient and intercept 
        [self.intercept, self.slope] = P.polyfit( (self.voltage_series/self.kV),
                                                    self.focus_series,
                                                    1,
                                                    full=False )
        # Covariance of two variables.
        self.cov = np.sqrt(np.diagonal(np.cov( (self.voltage_series/self.kV), 
                                                self.focus_series )))
        return


    # First order polynomial fitting using lmfit.
    def _fit_lmfit( self, **kwargs ):
        from lmfit import Model as mdl
        model = mdl( gradient_simple )
        params = model.make_params( )
        params['m'].value = -1.0
        params['m'].vary = True
        params['m'].max = 2
        params['c'].value = 0
        params['c'].vary = True
        self.results = model.fit( self.focus_series, params,
            x = (self.voltage_series/self.kV) )
        return


    ### Functions for printing results of fitting.
    def print_results( self, **kwargs ):
        '''
        Print results of lmfit fitting.
        '''
        method = kwargs.get( 'method', 'numpy' )
        if ( method == 'numpy' ):
            self._print_simple()
        if ( method == 'lmfit' ):
            self._print_lmfit()
        return


    # print results of simple fitting
    def _print_simple( self ):
        print( "Cc (mm): " + str( self.slope * 1e3 ) )
        return


    # print results of lmfit fitting
    def _print_lmfit( self ):
        print( self.results.fit_report( show_correl=False ) )
        print( 'Cc (mm): ' + str( self.results.params['m'].value * 1e3 ) + ' mm' )
        return

    
    ## Methods for plotting fit results.
    def plot_figure( self, **kwargs ):
        '''
        Plot results of fitting.

        Parameters
        ----------
        method : string, optional
            'simple' for numpy, 'lmfit' for lmfit
        '''
        method = kwargs.get( 'method', 'numpy' )
        if ( method == 'numpy' ):
            fig, ax = self._figure_simple()
        if ( method == 'lmfit' ):
            fig, ax = self._figure_lmfit()
        return fig, ax


    # Plot results of polyfit fitting.
    def _figure_simple( self ):
        '''
        Plot results of polynomial fitting.
        '''
        fig, ax = plt.subplots( )
        self._plot_data( fig, ax )
        ax.plot( (self.voltage_series/self.kV), 
                  gradient_simple( 
                    (self.voltage_series/self.kV),self.slope, self.intercept), 
                  color='red' )
        self._plot_labels( fig, ax )
        return fig, ax


    # Plot results of lmfit method.
    def _figure_lmfit( self ):
        '''
        Plot results of lmfit fitting.
        '''
        fig, ax = plt.subplots( )
        self._plot_data( fig, ax )
        ax.plot( (self.voltage_series/self.kV), self.results.best_fit,
            label='best fit', color='orange' )
        self._plot_labels( fig, ax )
        return fig, ax


    # Scatter plot of x and y data.
    def _plot_data( self, fig, ax ):
        ax.plot( (self.voltage_series/self.kV), self.focus_series, "x" )
        return


    # Add labels to figure.
    def _plot_labels( self, fig, ax ):
        ax.set_ylabel( "Defocus / m" )
        ax.set_xlabel( "V/V$_0$" )
        ytickslabels = ax.get_yticks()
        ax.set_yticks(ax.get_yticks(), ytickslabels*1e9)
        ax.grid( )
        return