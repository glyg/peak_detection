from .detection import detect_peaks
from .detection import NoPeaksDetectedException
from .detection import CanceledByUserException

import os
import logging

from color_system import color

logformat = color('%(asctime)s', 'BLUE') + ':'
logformat += color('%(levelname)s', 'RED') + ':'
logformat += color('%(name)s', 'YELLOW') + ':'
# logformat += color('%(funcName)s', 'GREEN') + ': '
logformat += color(' %(message)s', 'ENDC')

thisdir = os.path.abspath(os.path.dirname(__file__))
pkgdir = os.path.dirname(thisdir)
samplesdir = os.path.join(pkgdir, 'samples')

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(logformat, "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import warnings
warnings.filterwarnings("ignore")
