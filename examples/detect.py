"""
"""

import sys
sys.path.append("..")

from peak_detection import detect_peaks

fname = 'sample.tif'
peaks = detect_peaks(fname, parallel=True)
print(peaks.head(40))
