Gaussian Peak detection algorithm
=================================

Python implementation of the Gaussian peak detection described in
[Segré et al. Nature Methods (2008)](http://www.nature.com/nmeth/journal/v5/n8/full/nmeth.1233.html).

Dependence
----------

- numpy
- scipy
- scikit-image
- pandas

Usage example
--------------

```python
from peak_detection import detect_peaks
fname = 'sample.tif'
peaks = detect_peaks(fname, parallel=True)
```

```
2013-06-27 09:30:43:INFO:peak_detection.detection: Find peaks in sample.tif
2013-06-27 09:30:43:INFO:peak_detection.detection: Parallel mode enabled: 17 cores will be used to process 3 frames
2013-06-27 09:30:43:INFO:peak_detection.detection: Processing frame number 1/3 (33%)
2013-06-27 09:30:43:INFO:peak_detection.detection: Processing frame number 2/3 (66%)
2013-06-27 09:30:43:INFO:peak_detection.detection: Processing frame number 3/3 (100%)
```

```python
print(peaks.head(40))
```

```
                   x          y         w           I
stacks id
1      0   30.387653  49.136714  2.665996  137.964103
       1   41.013507  54.666204  2.639604   46.380415
2      0   30.341491  48.690070  2.718456  116.259774
       1   41.030425  54.582963  3.803210  113.208641
       2   52.243015  60.549746  2.880907   72.346001
       3   64.161942  66.806074  2.837319   61.754538
```
