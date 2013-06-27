import sys
sys.path.append("..")

from peak_detection import detect_peaks

fname = 'sample.tif'

detection_parameters = {'w_s': 10,
                        'peak_radius': 4.,
                        'threshold': 60.,
                        'max_peaks': 10
                        }

peaks = detect_peaks(fname, parallel=True, **detection_parameters)

for id, p in peaks.groupby(level="stacks"):
    print p.shape[0]
