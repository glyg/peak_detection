# -*- coding: utf-8 -*-

"""
We are implementing here the Gaussian peak detection described in
Segré et al. Nature Methods **5**, 8 (2008).[1]_

.. [1] http://dx.doi.org/10.1038/nmeth.1233
"""

from __future__ import division

import sys
import os
import logging
import multiprocessing
import itertools

import numpy as np
import pandas as pd

from scipy.optimize import leastsq
from skimage import feature

from peak_detection import in_ipython
from .progress import pprogress

log = logging.getLogger(__name__)


class NoPeaksDetectedException(Exception):
    pass


class CanceledByUserException(Exception):
    pass


class WrongArrayDimensions(Exception):
    pass


def detect_peaks(array,
                 shape_label=('t', 'z', 'x', 'y'),
                 parallel=True,
                 show_progress=False,
                 verbose=True,
                 **detection_parameters):
    """
    Sample must be a numpy array with at least 2 dimensions (x and y)
    """

    if array.ndim != len(shape_label):
        raise WrongArrayDimensions(
            'Array dimensions must fit with shape_label length arguments')

    if not verbose:
        log.disabled = True

    if not detection_parameters:
        detection_parameters = {'w_s': 10,
                                'peak_radius': 3.,
                                'threshold': 27.,
                                'max_peaks': 1e4
                                }

    peaks = find_stack_peaks(array,
                             shape_label,
                             parallel=parallel,
                             verbose=verbose,
                             show_progress=show_progress,
                             **detection_parameters)

    n_stacks = peaks.index.get_level_values('stacks').unique().size
    n_peaks = peaks.shape[0]
    log.info('%i peaks detected in %i stacks' % (n_peaks, n_stacks))

    return peaks


def find_stack_peaks(stacks,
                     shape_label,
                     parallel=False,
                     verbose=True,
                     show_progress=False,
                     **detection_parameters):
    """
    Finds the Gaussian peaks for all the pages in a multipage tif.

    Parameters
    ----------
    stacks : a TiffFile instance from Christoph Groethke's tifffile.py
    parallel : boolean
        If parallel != True, peak_detection try to use several core to process
        pages.

    Return
    -------
    masked_peaks : masked array
         The returned array contains the detected peaks for each 2D page.
         masked_array is a (N, M, 4) array, where N is the number of pages
          in the input file and M the maximum number of detected peaks in a page.
         For each peak (x, y, w, I) are given, where (x, y) is the position
         in the image, w the width of the gaussian spot and I the
         (background corrected) intensity.

    See Also
    --------
    find_gaussian_peaks for details on the detection method
    """

    # Array shape preprocessing: work only with 3 dimensions array
    original_shape = stacks.shape
    if len(original_shape) == 4:
        stacks = stacks.reshape((-1, ) + original_shape[-2:])
        shape_name = shape_label[:2]
    else:
        shape_name = shape_label[:1]

    # Check wether function is run from ipython and disable parallel feature if
    # is the case because python multiprocessing module is not yet compatible
    # with ipython.
    # if in_ipython():
        # parallel = False
    nb_stacks = len(stacks)

    if parallel:

        # Snippet to allow multiprocessing while importing
        # module such as numpy (only needed on linux)
        if os.name == 'posix':
            os.system("taskset -p 0xff %d" % os.getpid())

        def init_worker():
            import signal
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        ncore = multiprocessing.cpu_count() + 1
        log.info('Parallel mode enabled: %i cores will be used to process %i stacks' %
                 (ncore, nb_stacks))
        pool = multiprocessing.Pool(processes=ncore, initializer=init_worker)

    all_peaks = []
    i = 0

    # Build arguments list
    arguments = itertools.izip(
        stacks, itertools.repeat(detection_parameters), range(nb_stacks))

    try:
        # Launch peak_detection
        if parallel:
            results = pool.imap_unordered(find_gaussian_peaks, arguments)
        else:
            results = itertools.imap(find_gaussian_peaks, arguments)

        # Get unordered results and log progress
        for i in range(nb_stacks):
            result = results.next()
            if show_progress:
                pprogress((i + 1) / nb_stacks * 100, "%i peaks on stack %i" % (len(result[1]), result[0]))
            elif verbose:
                log.info('Detection done for stack number %i: %i peaks detected (%i/%i - %i%%)' %
                     (result[0], len(result[1]), i + 1, nb_stacks, ((i + 1) * 100 / nb_stacks)))
            all_peaks.append(result)

        if show_progress:
            pprogress(-1)

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise CanceledByUserException(
            'Peak detection has been canceled by user')

    # Sort peaks and remove index used to sort
    log.info('Reordering stacks')
    all_peaks.sort(key=lambda x: x[0])
    all_peaks = [x[1] for x in all_peaks]

    # Pre processing peaks
    index = []
    peaks = []

    for t, peak in enumerate(all_peaks):
        if peak.any():
            for i, p in enumerate(peak):
                index.append((t, i))
                peaks.append(p)

    if not peaks:
        raise NoPeaksDetectedException

    peaks = pd.DataFrame(peaks, columns=['x', 'y', 'w', 'I'], dtype='float')
    peaks.index = pd.MultiIndex.from_tuples(index, names=['stacks', 'id'])

    # Reshape DataFrame to fit with original_shape
    # Shape is given in DataFrame columns
    log.info('Add original shape to DataFrame as columns. Shape = %s' %
             str(original_shape))

    if len(original_shape) == 4:
        index = list(np.ndindex(original_shape[:2]))
    else:
        index = list(np.ndindex(original_shape[:1]))

    stacks_id = peaks.index.get_level_values('stacks')

    i_name = shape_label[0]
    i_col = map(lambda x: index[x][0], stacks_id)
    peaks[i_name] = i_col

    if len(original_shape) == 4:
        j_name = shape_label[1]
        j_col = map(lambda x: index[x][1], stacks_id)
        peaks[j_name] = j_col

    log.info('Detection is done')
    return peaks


def find_gaussian_peaks(args):
    """
    Buffer function for _find_gaussian_peaks
    """
    frame, detection_parameters, i = args
    return (i, _find_gaussian_peaks(frame, **detection_parameters))


def _find_gaussian_peaks(image, w_s=15, peak_radius=1.5,
                         threshold=27., max_peaks=1e4):
    """
    This function implements the Gaussian peak detection described
    in Segré et al. Nature Methods **5**, 8 (2008). It is based on a
    likelyhood ratio test on the presence or absence of a Gaussian
    peak on a patch moving over the input 2D image and a successive
    sub-pixel localization of the peaks by a least square fit.  This
    detection is followed recursively by a _deflation_ of the image
    from the detected peaks and a new detection, until no more peaks
    are found

    Parameters
    ----------

    image: a 2D array
        the input image
    w_s: int, optional
        the width of the sliding window over which the hypothesis ratio
        is computed :math:`w_s` in the article. It should be wide enough
        to contain some background to allow a proper noise evaluation.
    peak_radius: float, optional
        typical radius of the peaks to detect. It must be higher than one
        (as peaks less that a pixel wide would yield bogus results
    thershold: float, optional
        Criterium for a positive detection (i.e. the null hypothesis is false).
        Corresponds to the :mat:`\chi^2` parameter in the Constant False
        Alarm Rate section of the article supplementary text (p. 12).
        A higher `threshold` corresponds to a more stringent test.
        According to the authors, this parameters needs to be adjusted
        once for a given data set.

    Returns
    -------

    peaks: ndarray
        peaks is a Nx4 array, where N is the number of detected peaks in the
        image. Each line gives the x position, y position, width,
        and (background corrected) intensity of a detected peak (in that order).

    """
    peaks_coords = glrt_detection(image, peak_radius,
                                  w_s, threshold)
    peaks = gauss_estimation(image, peaks_coords, w_s)
    d_image = image_deflation(image, peaks, w_s)
    peaks_coords = glrt_detection(d_image, peak_radius,
                                  w_s, threshold)
    while len(peaks_coords) > 0 and len(peaks) < max_peaks:
        new_peaks = gauss_estimation(d_image, peaks_coords, w_s)
        # in case the 2D gauss fit fails
        if len(new_peaks) < 1:
            break
        peaks.extend(new_peaks[:])
        d_image = image_deflation(d_image, new_peaks, w_s)
        peaks_coords = glrt_detection(d_image, peak_radius,
                                      w_s, threshold)
    peaks = np.array(peaks)
    return peaks


def image_deflation(image, peaks, w_s):
    """
    Substracts the detected Gaussian peaks from the input image and
    returns the deflated image.
    """
    d_image = image.copy()
    for peak in peaks:
        xc, yc, width, I = peak
        xc_rel = w_s // 2 + xc - np.floor(xc)
        yc_rel = w_s // 2 + yc - np.floor(yc)
        low_x = int(xc - w_s // 2)
        low_y = int(yc - w_s // 2)

        params = xc_rel, yc_rel, width, I, 0
        deflated_peak = gauss_continuous(params, w_s)
        d_image[low_x:low_x + w_s,
                low_y:low_y + w_s] -= deflated_peak.reshape((w_s, w_s))
    return d_image


def gauss_estimation(image, peaks_coords, w_s):
    """
    Least square fit of a 2D Gauss peaks (with radial symmetry)
    on regions of width `w_s` centered on each element
    of `peaks_coords`.

    Parameters:
    ----------
    image : 2D array
        a greyscale 2D input image.
    peaks_coords: iterable of pairs of int.
       The peaks_coords should contain `(x, y)` pairs
       corresponding to the approximate peak center,
       in pixels.
    """
    peaks = []
    for coords in peaks_coords:
        low_x, low_y = coords - w_s // 2
        try:
            patch = image[low_x: low_x + w_s,
                          low_y: low_y + w_s]
            params, success = gauss_estimate(patch, w_s)
            xc, yc, width, I, bg = params
            if success and I > 0 and width < w_s:
                peaks.append([xc + low_x, yc + low_y, width, I])
        except IndexError:
            log.error('peak too close from the edge\n'
                      'use a smaller window\n'
                      'peak @ (%i, %i) discarded' % (coords[0], coords[1]))
            continue
    return peaks


def glrt_detection(image, r0, w_s, threshold):
    """
    Implements the Generalized Likelyhood Ratio Test, by
    computing equation 4 in Segré et al. Supplementary Note (p. 12)
    in a window sliding other the image.

    Parameters:
    ----------
    image: array
        the 2D input image
    r0: float
        the detected Gaussian peak 1/e radius
    w_s: int
        Size of the sliding window over which the test is
        computed ( :math:`w_s` in the article).
    threshold: float
        Criterium for a positive detection (i.e. the null hypothesis is false).
        Corresponds to the :mat:`\chi^2` parameter in the Constant False
        Alarm Rate section of the article supplementary text (p. 12).
        A higher `threshold` corresponds to a more stringent test.
        According to the authors, this parameters needs to be adjusted
        once for a given data set.

    Returns:
    --------

    peaks_coords: array
        An Nx2 array containing (x, y) pairs of the detected peaks
        in integer pixel coordinates.
    """
    w, h = image.shape
    g_patch = gauss_patch(r0, w_s)
    g_patch -= g_patch.mean()
    g_squaresum = np.sum(g_patch ** 2)
    hmap = np.array([hypothesis_map(image[i: i + w_s, j: j + w_s],
                                    g_patch, g_squaresum)
                     for i, j in np.ndindex((w - w_s, h - w_s))])
    hmap = -2 * hmap.reshape((w - w_s, h - w_s))
    peaks_coords = feature.peak_local_max(hmap, 3,
                                          threshold_abs=threshold)
    peaks_coords += w_s // 2
    return peaks_coords


def hypothesis_map(patch, g_patch, g_squaresum):
    """
    Computes the ratio for a given patch position.
    """
    w_s = patch.shape[0]
    # mean = patch.mean()
    multiplicative = g_patch * patch

    intensity = multiplicative.sum()
    normalisation = w_s * patch.std()
    ratio = (w_s ** 2 / 2.) * np.log(1 - (intensity
                                          / normalisation) ** 2
                                     / g_squaresum)
    return ratio


def gauss_estimate(patch, w_s):
    """
    Least square 2D gauss fit
    """
    params0 = [w_s / 2., w_s / 2., 3.,
               np.float(patch.max() - patch.min()), np.float(patch.min())]
    errfunc = lambda p: patch.flatten() - gauss_continuous(p, w_s)
    return leastsq(errfunc, params0, xtol=0.01)


def gauss_continuous(params, w_s):
    """2D gauss function with a float center position"""
    xc, yc, width, I, bg = params
    xc = np.float(xc)
    yc = np.float(yc)
    x = np.exp(- (np.arange(0, w_s) - xc) ** 2 / width ** 2)
    y = np.exp(- (np.arange(0, w_s) - yc) ** 2 / width ** 2)
    g_patch = I * np.outer(x, y) + bg
    return g_patch.flatten()


def gauss_patch(r0, w_s):
    """
    Computes an w_s by w_s image with a
    power normalized  Gaussian peak with radial symmetry
    at its center.
    """
    x = y = np.exp(- (np.arange(w_s) - w_s // 2) ** 2 / r0 ** 2)
    A = 1. / (np.sqrt(np.pi) * r0)
    g_patch = A * np.outer(x, y)
    return g_patch


def gauss_discrete(r0, i, j, w_s):
    """
    2D gauss function with a discrete center position
    """
    i -= w_s // 2
    j -= w_s // 2
    A = 1. / (np.sqrt(np.pi) * r0)
    return A * np.exp(-(i ** 2 + j ** 2) / r0 ** 2)
