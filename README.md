Gaussian Peak detection algorithm
=================================

Python implementation of the Gaussian peak detection described in Segré et al. Nature Methods (2008).

Dependence
----------

numpy
scipy
scikit-image
pandas

Usage example
--------------


    from peak_detection import detect_peaks
    fname = 'sample.tif'
    peaks = detect_peaks(fname, parallel=True)
    print(peaks.head(40))
