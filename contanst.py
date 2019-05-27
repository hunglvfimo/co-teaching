import numpy as np
import os

TRIPLET_MARGIN = 0.1

DATA_DIR = os.path.join('..', '..', 'data', 'processed')
MODEL_DIR = os.path.join('..', '..', 'models')

MEAN_SAR_4L = (0.0625, 0.0625, 0.0625)
STD_SAR_4L = (0.0784, 0.0784, 0.0784)
CLASSES_SAR_4L = ['Cargo', 'Other Type', 'Tanker', 'Tug']
NUM_SAR_4L = np.array([1587.0, 136.0, 231.0, 41.0])

MEAN_SAR_8A = (0.0625, 0.0625, 0.0625)
STD_SAR_8A = (0.0784, 0.0784, 0.0784)
CLASSES_SAR_8A = ['Cargo', 'Dredging', 'Fishing', 'Other Type', 'Passenger', 'Search', 'Tanker', 'Tug']
NUM_SAR_8A = np.array([1587.0, 11.0, 7.0, 136.0, 4.0, 6, 231.0, 41.0])

MEAN_VAIS_RGB = [0.3276, 0.4221, 0.5979]
STD_VAIS_RGB = [0.2144, 0.2404, 0.2955]
CLASSES_VAIS_RGB = ['cargo', 'medium-other', 'passenger', 'sailing', 'small', 'tug']
NUM_VAIS_RGB = np.array([100, 99, 78, 203, 323, 37])

MEAN_G_FLOOD = (0.4791, 0.4798, 0.4630)
STD_G_FLOOD = (0.2539, 0.2505, 0.2693)
CLASSES_G_FLOOD = ['flood', 'non-flood']
NUM_G_FLOOD = np.array([1713, 1729])

MEAN_MEDEVAL17 = (0.4589, 0.4623, 0.4339)
STD_MEDEVAL17 = (0.2455, 0.2466, 0.2756)
CLASSES_MEDEVAL17 = ['flood', 'non-flood']
NUM_MEDEVAL17 = np.array([1728, 3042])