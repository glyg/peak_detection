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
detection_parameters = {'w_s': 10,
                        'peak_radius': 4.,
                        'threshold': 60.,
                        'max_peaks': 10
                        }
peaks = detect_peaks(fname, parallel=True, **detection_parameters)
```

```
2013-06-27 19:53:13:INFO:peak_detection.detection: Find peaks in /home/hadim/Insync/Documents/phd/dev/peak_detection/examples/sample.tif
2013-06-27 19:53:13:INFO:peak_detection.detection: Parallel mode enabled: 5 cores will be used to process 9 stacks
2013-06-27 19:53:14:INFO:peak_detection.detection: Detection done for stack number 1: 2 peaks detected (1/9 - 11%)
2013-06-27 19:53:14:INFO:peak_detection.detection: Detection done for stack number 3: 4 peaks detected (2/9 - 22%)
2013-06-27 19:53:15:INFO:peak_detection.detection: Detection done for stack number 0: 1 peaks detected (3/9 - 33%)
2013-06-27 19:53:15:INFO:peak_detection.detection: Detection done for stack number 2: 3 peaks detected (4/9 - 44%)
2013-06-27 19:53:16:INFO:peak_detection.detection: Detection done for stack number 4: 5 peaks detected (5/9 - 55%)
2013-06-27 19:53:16:INFO:peak_detection.detection: Detection done for stack number 5: 6 peaks detected (6/9 - 66%)
2013-06-27 19:53:16:INFO:peak_detection.detection: Detection done for stack number 7: 8 peaks detected (7/9 - 77%)
2013-06-27 19:53:17:INFO:peak_detection.detection: Detection done for stack number 6: 7 peaks detected (8/9 - 88%)
2013-06-27 19:53:17:INFO:peak_detection.detection: Detection done for stack number 8: 9 peaks detected (9/9 - 100%)
2013-06-27 19:53:17:INFO:peak_detection.detection: Reordering stacks
2013-06-27 19:53:17:INFO:peak_detection.detection: Add original shape to DataFrame as columns. Shape = (3, 3, 54, 209)
2013-06-27 19:53:17:INFO:peak_detection.detection: Detection is done
2013-06-27 19:53:17:INFO:peak_detection.detection: 45 peaks detected in 9 stacks
```

```python
print(peaks)
```

```
                   x           y       w           I  t  z
stacks id
0      0   19.022877   28.102197  2.9038  195.396065  0  0
1      0   19.022877   28.102197  2.9038  195.396065  0  1
       1   25.022877  189.102197  2.9038  195.396065  0  1
2      0   14.022877  133.102197  2.9038  195.396065  0  2
       1   29.022877   44.102197  2.9038  195.396065  0  2
       2   29.022877   97.102197  2.9038  195.396065  0  2
3      0   19.022877   28.102197  2.9038  195.396065  1  0
       1   24.022877  132.102197  2.9038  195.396065  1  0
       2   24.022877  178.102197  2.9038  195.396065  1  0
       3   29.022877   79.102197  2.9038  195.396065  1  0
4      0   19.022877   27.102197  2.9038  195.396065  1  1
       1   26.022877  181.102197  2.9038  195.396065  1  1
       2   28.022877   80.102197  2.9038  195.396065  1  1
       3   28.022877  128.102197  2.9038  195.396065  1  1
       4   45.022877   43.102197  2.9038  195.396065  1  1
5      0   15.022877  147.102197  2.9038  195.396065  1  2
       1   17.022877   55.102197  2.9038  195.396065  1  2
       2   18.022877   88.102197  2.9038  195.396065  1  2
       3   27.022877   22.102197  2.9038  195.396065  1  2
       4   35.022877  122.102197  2.9038  195.396065  1  2
       5   38.022877   66.102197  2.9038  195.396065  1  2
6      0   14.022877  131.102197  2.9038  195.396065  2  0
       1   15.022877   75.102197  2.9038  195.396065  2  0
       2   32.022877   39.102197  2.9038  195.396065  2  0
       3   34.022877   99.102197  2.9038  195.396065  2  0
       4   36.022877   67.102197  2.9038  195.396065  2  0
       5   36.022877  157.102197  2.9038  195.396065  2  0
       6   37.022877  125.102197  2.9038  195.396065  2  0
7      0   14.022877  131.102197  2.9038  195.396065  2  1
       1   15.022877   75.102197  2.9038  195.396065  2  1
       2   16.022877  176.102197  2.9038  195.396065  2  1
       3   32.022877   39.102197  2.9038  195.396065  2  1
       4   34.022877   99.102197  2.9038  195.396065  2  1
       5   36.022877   67.102197  2.9038  195.396065  2  1
       6   36.022877  157.102197  2.9038  195.396065  2  1
       7   37.022877  125.102197  2.9038  195.396065  2  1
8      0   14.022877    8.102197  2.9038  195.396065  2  2
       1   14.022877  131.102197  2.9038  195.396065  2  2
       2   15.022877   75.102197  2.9038  195.396065  2  2
       3   16.022877  176.102197  2.9038  195.396065  2  2
       4   32.022877   39.102197  2.9038  195.396065  2  2
       5   34.022877   99.102197  2.9038  195.396065  2  2
       6   36.022877   67.102197  2.9038  195.396065  2  2
       7   36.022877  157.102197  2.9038  195.396065  2  2
       8   37.022877  125.102197  2.9038  195.396065  2  2

```
