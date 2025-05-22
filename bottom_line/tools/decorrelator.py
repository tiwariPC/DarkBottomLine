import pandas as pd
import numpy as np
import pickle as pkl
import gzip
import numba
import scipy.optimize as opt


class decorrelator(object):

    def __init__(self, df, var, dvar, bins, method='binned', verbose=False):

        self.df = df.loc[:, [var, dvar]]
        self.df.reset_index(inplace=True)
        self.bins = bins
        self.xc = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.var = var
        self.dvar = dvar
        self.verbose = verbose
        if method not in ['binned', 'unbinned']:
            print('WARNING: Method not implemented!')
        self.method = method

    def loadCdfs(self, path):

        with gzip.open(path) as f:
            self.cdfs = pkl.load(f)

        # if not isinstance(self.cdfs[list(self.cdfs.keys())[0]], np.array):
        #     raise TypeError('Cdfs need to be numpy arrays of shape (2,nBins)')

        # if self.method == 'binned' and 'scipy.interpolate' not in getattr(self.cdfs[list(self.cdfs.keys())[0]], '__module__', None):
        #     raise TypeError('If decorrelation is done binned, cdfs need to be submodule of scipy.interpolate')

    def correctY_evt(self, val, cdf, cdf_ref):

        if np.searchsorted(cdf[1],val) >= len(cdf[0]):
            cum_mc = 1
        else:
            cum_mc = cdf[0][np.searchsorted(cdf[1],val)]

        if cum_mc == 1 or np.searchsorted(cdf_ref[0], cum_mc) >= len(cdf_ref[0]):
            return cdf_ref[1][-1]
        else:
            return cdf_ref[1][np.searchsorted(cdf_ref[0], cum_mc)]

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def correctY_binned(arr, cdf, cdf_ref):

        # this operation take two parts, first calculates the values of the CDF o y (variable to be corrected)
        # since CDF = a*y + b and mkaing some calculation, interpolating betwenn two points
        # CDF = (Delta CDF)/(Delta Y)*(y - ya) + CDF a' (notation kind of meh, look at my notes!)

        # when the CDF(y) is calculated, we just use CDF(y') = CDF(y) -> y' = CDF-1(CDF(y)) and vouala!

        indCdf = np.searchsorted(cdf[1], arr)
        cdfVal = ((cdf[0][indCdf] - cdf[0][indCdf - np.ones_like(indCdf)]) / (cdf[1][indCdf] - cdf[1][indCdf - 1])) * (arr - cdf[1][indCdf - np.ones_like(indCdf)]) + cdf[0][indCdf - np.ones_like(indCdf)]
        indCorr = np.searchsorted(cdf_ref[0], cdfVal)
        corrVal = ((cdf_ref[1][indCorr] - cdf_ref[1][indCorr - np.ones_like(indCorr)]) / (cdf_ref[0][indCorr] - cdf_ref[0][indCorr - np.ones_like(indCorr)])) * (cdfVal - cdf_ref[0][indCorr - np.ones_like(indCorr)]) + cdf_ref[1][indCorr - np.ones_like(indCorr)]
        # ind = np.searchsorted(cdf[1], arr)
        # one = np.ones_like(ind)
        # return ((cdf_ref[1][ind] - cdf_ref[1][ind - one])/(cdf[1][ind] - cdf[1][ind - one])) * (arr - cdf[1][ind - one]) + cdf_ref[1][ind - one]
        return corrVal

    def quantMorphInterp(self, val, IntCdf, IntCdfRef):

        cdfVal = IntCdf(val)

        if cdfVal == 0. or cdfVal > 0.999:
            return val
        if cdfVal < 0.:
            print(cdfVal)

        corrVal = opt.root_scalar(lambda x: IntCdfRef(x) - cdfVal,bracket=[0.,1.], method='brenth')
        return corrVal.root

    def correctY_arr(self, arr, cdf, cdf_ref):
        if self.method == 'unbinned':
            return np.array([self.correctY_evt(val, cdf, cdf_ref) for val in arr])
        elif self.method == 'binned':
            return self.correctY_binned(arr, cdf, cdf_ref)
            # return np.array([self.quantMorphInterp(val, cdf, cdf_ref) for val in arr])

    def findMassBin(self, mass):

        ind = np.searchsorted(self.bins, mass, side='right')
        return self.xc[ind - 1]

    def findGb(self):

        self.df['{}_bin'.format(self.dvar)] = pd.cut(self.df[self.dvar].values, bins=self.bins, labels=[str(x) for x in self.xc])
        self.gb = self.df.groupby('{}_bin'.format(self.dvar), observed=False)

    def doDecorr(self, ref):

        self.findGb()
        cdf_ref = self.cdfs[str(self.findMassBin(ref))]
        for name, grp in self.gb:
            self.verbose = False
            if self.verbose:
                print('--------------------------------------------------------------------------------')
                print('Decorrelating for mass {}'.format(name))
                print('Number of events in mass bin {}'.format(grp.index.size))
            cdf = self.cdfs[name]
            self.df.loc[grp.index, '{}_decorr'.format(self.var)] = self.correctY_binned(grp[self.var].values, cdf, cdf_ref)

        return self.df['{}_decorr'.format(self.var)].values


class cdfCalc(decorrelator):

    def __init__(self, df, var, dvar, bins, method='binned', dBins=np.linspace(0.,0.5,1001), weightstr='weight'):

        super(cdfCalc, self).__init__(df, var, dvar, bins, method=method)
        self.df[weightstr] = df[weightstr]
        self.weightstr = weightstr
        self.dBins = dBins

        self.findGb()

    @staticmethod
    def _calcCdf(val, weights):

        df = pd.DataFrame(data=np.vstack((val, weights)).T, columns=['val', 'weights'])
        df.sort_values('val', inplace=True)
        w_cum = np.cumsum(df['weights'].values)
        w_cum /= w_cum[-1]

    def _calcCdfBinned(self, val, weights):

        hist, _ = np.histogram(val, weights=weights, bins=self.dBins)
        rightEdge = self.dBins[1:]
        bCum = np.cumsum(hist)
        bCum[bCum < 0.] = 0
        bCum /= bCum.max()
        cdfBinned = np.vstack((bCum,rightEdge))

        # Make a plot of this cdfBinned here!
        return cdfBinned  # interp.PchipInterpolator(cdfBinned[1], cdfBinned[0])

    def calcCdfs(self):

        """
        For each mass bin (chosen in 01_dumpCdfs_decorr), it calls _caldCdfBinned, which histograms the sigma_m_over_m and calculates its CDFs which is later saved to a file.
        """

        self.cdfs = {}
        if self.method == 'unbinned':
            for key in self.gb.groups.keys():
                self.cdfs[key] = self._calcCdf(self.gb.get_group(key)[self.var].values, self.gb.get_group(key)[self.weightstr].values)
        elif self.method == 'binned':
            for key in self.gb.groups.keys():
                self.cdfs[key] = self._calcCdfBinned(self.gb.get_group(key)[self.var].values, self.gb.get_group(key)[self.weightstr].values)

    def dumpCdfs(self, ofile):

        if not hasattr(self, 'cdfs'):
            self.calcCdfs()

        with gzip.open(ofile, 'w') as f:
            pkl.dump(self.cdfs, f)
